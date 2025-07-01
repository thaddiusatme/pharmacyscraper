"""
Comprehensive tests for the Classifier class.
"""
import pytest
from unittest.mock import patch, MagicMock

from pharmacy_scraper.classification.classifier import Classifier, _classification_cache
from pharmacy_scraper.classification.models import (
    PharmacyData,
    ClassificationResult,
    ClassificationSource,
    ClassificationMethod,
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient


@pytest.fixture
def mock_perplexity_client():
    """Provides a mocked PerplexityClient instance that returns a high-confidence LLM result."""
    mock_client = MagicMock(spec=PerplexityClient)
    mock_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=True,
        is_compounding=False,
        confidence=0.9,
        source=ClassificationSource.PERPLEXITY,
        reason="Mocked LLM result",
        method=ClassificationMethod.LLM,
    )
    return mock_client


@pytest.fixture
def sample_pharmacy_data():
    """Provides a consistent sample PharmacyData object for tests."""
    return PharmacyData(name="HealthFirst Pharmacy", address="123 Wellness Ave")


@pytest.fixture(autouse=True)
def isolated_cache():
    """Ensures each test runs with a clean, isolated cache."""
    _classification_cache.clear()
    yield
    _classification_cache.clear()


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_high_confidence_rule_bypasses_llm(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that a high-confidence rule-based result is returned directly without calling the LLM."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=True, confidence=0.95, source=ClassificationSource.RULE_BASED
    )
    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data)

    assert result.source == ClassificationSource.RULE_BASED
    mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_not_called()


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_low_confidence_rule_triggers_llm(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that a low-confidence rule-based result triggers an LLM fallback."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=False, confidence=0.5, source=ClassificationSource.RULE_BASED
    )
    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data)

    assert result.source == ClassificationSource.PERPLEXITY
    mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_called_once_with(sample_pharmacy_data)


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_llm_disabled_returns_rule_result(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that the rule-based result is returned when `use_llm=False`, regardless of confidence."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=False, confidence=0.5, source=ClassificationSource.RULE_BASED
    )
    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data, use_llm=False)

    assert result.source == ClassificationSource.RULE_BASED
    mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_not_called()


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_llm_error_falls_back_to_rule_result(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that the rule-based result is returned if the LLM call fails."""
    rule_result = ClassificationResult(
        is_chain=False, confidence=0.5, source=ClassificationSource.RULE_BASED
    )
    mock_rule_based.return_value = rule_result
    mock_perplexity_client.classify_pharmacy.side_effect = Exception("API Error")
    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data)

    assert result == rule_result
    mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_called_once_with(sample_pharmacy_data)


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_caching_behavior(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that results are cached and subsequent calls do not trigger new classifications."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=False, confidence=0.5, source=ClassificationSource.RULE_BASED
    )
    classifier = Classifier(client=mock_perplexity_client)

    # First call, should trigger rule-based and LLM, and cache the result
    classifier.classify_pharmacy(sample_pharmacy_data)
    mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_called_once_with(sample_pharmacy_data)

    # Second call, should hit the cache and not call the mocks again
    result = classifier.classify_pharmacy(sample_pharmacy_data)
    assert result.method == ClassificationMethod.CACHED
    mock_rule_based.assert_called_once()  # Not called again
    mock_perplexity_client.classify_pharmacy.assert_called_once()  # Not called again


def test_invalid_input_handling():
    """Tests that the classifier raises a ValueError for None input."""
    classifier = Classifier(client=None)
    with pytest.raises(ValueError, match="Pharmacy data cannot be None"):
        classifier.classify_pharmacy(None)
