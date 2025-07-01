"""
Tests for the pharmacy classification module, focusing on the Classifier class.
"""
"""
Tests for the pharmacy classification module, focusing on the Classifier class.
"""
import pytest
from unittest.mock import patch, MagicMock

from pharmacy_scraper.classification.classifier import (
    Classifier,
    _classification_cache,
)
from pharmacy_scraper.classification.models import (
    PharmacyData,
    ClassificationResult,
    ClassificationSource,
    ClassificationMethod,
)
from pharmacy_scraper.classification.perplexity_client import PerplexityClient


@pytest.fixture
def mock_perplexity_client():
    """Provides a robust mocked PerplexityClient instance with a spec."""
    mock_client = MagicMock(spec=PerplexityClient)
    mock_client.classify_pharmacy.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.8,
        source=ClassificationSource.PERPLEXITY,
        reason="Mocked LLM result",
        method=ClassificationMethod.LLM,
    )
    return mock_client


@pytest.fixture(autouse=True)
def isolated_cache():
    """Ensures each test runs with a clean, isolated cache."""
    _classification_cache.clear()
    yield
    _classification_cache.clear()


@pytest.fixture
def sample_pharmacy_data():
    """Provides a consistent sample PharmacyData object for tests."""
    return PharmacyData(
        name="HealthFirst Pharmacy",
        address="123 Wellness Ave, Suite 100, Healtheville, USA",
        phone="(123) 456-7890",
        website="http://www.healthfirst.com",
    )


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_classifier_instantiation_and_rule_based_fallback(
    mock_rule_based, sample_pharmacy_data
):
    """Tests that the Classifier can be instantiated without a client and falls back to rule-based classification."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=True,
        is_compounding=False,
        confidence=0.95,
        source=ClassificationSource.RULE_BASED,
        reason="Rule match",
        method=ClassificationMethod.RULE_BASED,
    )
    try:
        classifier = Classifier(client=None)
        result = classifier.classify_pharmacy(sample_pharmacy_data, use_llm=False)
        assert result.source == ClassificationSource.RULE_BASED
        mock_rule_based.assert_called_once_with(sample_pharmacy_data)
    except Exception as e:
        pytest.fail(f"Classifier instantiation or use failed unexpectedly: {e}")


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_llm_override(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that a low-confidence rule-based result is overridden by the LLM."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.5,
        source=ClassificationSource.RULE_BASED,
        reason="Low confidence rule",
        method=ClassificationMethod.RULE_BASED,
    )

    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data, use_llm=True)

    mock_perplexity_client.classify_pharmacy.assert_called_once()
    assert result.source == ClassificationSource.PERPLEXITY


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_caching_with_llm_result(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that results from the LLM are cached correctly."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.1,
        source=ClassificationSource.RULE_BASED,
        reason="Very low confidence",
        method=ClassificationMethod.RULE_BASED,
    )

    classifier = Classifier(client=mock_perplexity_client)

    classifier.classify_pharmacy(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_called_once()

    classifier.classify_pharmacy(sample_pharmacy_data)
    mock_perplexity_client.classify_pharmacy.assert_called_once()


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_no_llm_on_high_confidence_rule(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that the LLM is not called if the rule-based result has high confidence."""
    mock_rule_based.return_value = ClassificationResult(
        is_chain=True,
        is_compounding=False,
        confidence=1.0,
        source=ClassificationSource.RULE_BASED,
        reason="High confidence rule",
        method=ClassificationMethod.RULE_BASED,
    )

    classifier = Classifier(client=mock_perplexity_client)
    classifier.classify_pharmacy(sample_pharmacy_data, use_llm=True)

    mock_perplexity_client.classify_pharmacy.assert_not_called()


@patch("pharmacy_scraper.classification.classifier.rule_based_classify")
def test_llm_failure_fallback(
    mock_rule_based, mock_perplexity_client, sample_pharmacy_data
):
    """Tests that the classifier falls back to the rule-based result if the LLM fails."""
    rule_result = ClassificationResult(
        is_chain=False,
        is_compounding=False,
        confidence=0.1,
        source=ClassificationSource.RULE_BASED,
        reason="Failing over",
        method=ClassificationMethod.RULE_BASED,
    )
    mock_rule_based.return_value = rule_result
    mock_perplexity_client.classify_pharmacy.side_effect = Exception("API unavailable")

    classifier = Classifier(client=mock_perplexity_client)
    result = classifier.classify_pharmacy(sample_pharmacy_data, use_llm=True)

    assert result == rule_result
    assert result.source == ClassificationSource.RULE_BASED
