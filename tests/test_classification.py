"""
Tests for the pharmacy classification module.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from typing import Dict, Any

# Test data
SAMPLE_PHARMACIES = [
    {
        "name": "Downtown Pharmacy",
        "address": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94105",
        "phone": "(555) 123-4567"
    },
    {
        "name": "CVS Pharmacy #1234",
        "address": "456 Market St",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94103",
        "phone": "(555) 987-6543"
    },
    {
        "name": "Family Care Compounding",
        "address": "789 Oak St",
        "city": "Austin",
        "state": "TX",
        "zip": "73301",
        "phone": "(555) 456-7890"
    }
]

def test_llm_classification():
    """Test LLM-based classification of pharmacies."""
    # Skip if Perplexity API key is not available
    pytest.importorskip("perplexity")
    
    from src.classification import classify_pharmacy
    
    # Test independent pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[0])
    assert "is_chain" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1.0
    
    # Test chain pharmacy
    result = classify_pharmacy(SAMPLE_PHARMACIES[1])
    assert result["is_chain"] is True
    assert result["confidence"] >= 0.9  # High confidence for chains

def test_rule_based_classification():
    """Test rule-based classification fallback."""
    from src.classification import rule_based_classify
    
    # Test independent pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[0])
    assert result["is_chain"] is False
    assert result["method"] == "rule_based"
    
    # Test chain pharmacy
    result = rule_based_classify(SAMPLE_PHARMACIES[1])
    assert result["is_chain"] is True
    assert result["method"] == "rule_based"

def test_compounding_pharmacy_detection():
    """Test that compounding pharmacies are properly identified as independent."""
    from src.classification import rule_based_classify
    
    result = rule_based_classify(SAMPLE_PHARMACIES[2])
    assert result["is_chain"] is False
    assert "compounding" in result["reason"].lower()

def test_batch_classification():
    """Test batch processing of pharmacy classification."""
    pytest.importorskip("perplexity")
    from src.classification import batch_classify_pharmacies
    
    df = pd.DataFrame(SAMPLE_PHARMACIES)
    results = batch_classify_pharmacies(df)
    
    assert len(results) == len(SAMPLE_PHARMACIES)
    assert all(col in results.columns for col in ["is_chain", "confidence", "classification_method"])
    assert results["is_chain"].dtype == bool

def test_classification_with_mock():
    """Test classification with mocked LLM responses."""
    with patch("src.classification.classifier.query_perplexity") as mock_llm:
        # Mock LLM response for independent pharmacy
        mock_llm.return_value = {
            "is_chain": False,
            "confidence": 0.95,
            "reason": "Independent based on name and lack of chain indicators",
            "method": "llm"
        }
        
        from src.classification import classify_pharmacy
        result = classify_pharmacy(SAMPLE_PHARMACIES[0])
        
        assert result["is_chain"] is False
        assert result["confidence"] >= 0.9
        assert "reason" in result
        mock_llm.assert_called_once()

def test_confidence_threshold():
    """Test that low confidence results fall back to rule-based classification."""
    with patch("src.classification.classifier.query_perplexity") as mock_llm:
        # Mock low confidence LLM response
        mock_llm.return_value = {
            "is_chain": False,
            "confidence": 0.6,  # Below threshold
            "reason": "Not confident",
            "method": "llm"
        }
        
        from src.classification import classify_pharmacy
        result = classify_pharmacy(SAMPLE_PHARMACIES[0])
        
        # Should use rule-based fallback
        assert result["method"] == "rule_based"
