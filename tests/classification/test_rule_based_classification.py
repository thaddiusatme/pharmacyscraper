"""
Tests for rule-based classification functionality in the classifier module.

This test suite focuses on:
1. Chain pharmacy detection logic
2. Compounding pharmacy detection
3. Independent pharmacy classification
4. Edge cases in rule-based classification
"""
import pytest
from unittest.mock import patch, MagicMock

from pharmacy_scraper.classification.classifier import (
    rule_based_classify,
    CHAIN_IDENTIFIERS
)
from pharmacy_scraper.classification.models import (
    PharmacyData,
    ClassificationResult,
    ClassificationSource
)


class TestChainPharmacyDetection:
    """Tests for chain pharmacy detection in rule-based classification."""
    
    @pytest.mark.parametrize("chain_name", CHAIN_IDENTIFIERS)
    def test_all_chain_identifiers(self, chain_name: str) -> None:
        """Test that each chain identifier in CHAIN_IDENTIFIERS is detected.
        
        Args:
            chain_name: Chain pharmacy name from CHAIN_IDENTIFIERS list
        """
        # Create a pharmacy with the chain name
        pharmacy = PharmacyData(name=f"{chain_name} Pharmacy")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as a chain
        assert result.is_chain is True
        assert result.is_compounding is False
        assert result.confidence >= 0.9
        assert chain_name in result.explanation
        assert result.source == ClassificationSource.RULE_BASED
    
    def test_chain_with_multiple_identifiers(self) -> None:
        """Test pharmacy name with multiple chain identifiers."""
        # Create a pharmacy with multiple chain identifiers
        pharmacy = PharmacyData(name="CVS Pharmacy at Target")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as a chain
        assert result.is_chain is True
        # The explanation should mention one of the chains
        assert any(chain in result.explanation for chain in ["CVS", "Target"])
    
    def test_chain_with_suffix_only(self) -> None:
        """Test pharmacy with chain identifier only in suffix."""
        # Create a pharmacy with chain only in suffix
        pharmacy = PharmacyData(name="Downtown Walgreens")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as a chain
        assert result.is_chain is True
        assert "Walgreens" in result.explanation


class TestCompoundingPharmacyDetection:
    """Tests for compounding pharmacy detection in rule-based classification."""
    
    def test_compounding_in_name(self) -> None:
        """Test detection of compounding pharmacy from name."""
        pharmacy = PharmacyData(name="ABC Compounding Pharmacy")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as a compounding pharmacy
        assert result.is_chain is False
        assert result.is_compounding is True
        assert result.confidence >= 0.9
        assert "Compounding pharmacy" in result.explanation
        assert result.source == ClassificationSource.RULE_BASED
    
    def test_compounding_with_chain_identifier(self) -> None:
        """Test compounding pharmacy with chain identifier."""
        # This tests the order of rules - compounding check happens after chain check
        pharmacy = PharmacyData(name="Compounding CVS")
        result = rule_based_classify(pharmacy)
        
        # Chain detection should take precedence over compounding
        assert result.is_chain is True
        assert result.is_compounding is False
    
    def test_avoid_false_positives(self) -> None:
        """Test that partial matches don't trigger compounding detection."""
        # Using a completely different name without 'compounding' at all
        pharmacy = PharmacyData(name="Normal Pharmacy Services")
        result = rule_based_classify(pharmacy)
        
        # Should not be detected as compounding
        assert result.is_compounding is False
    
    def test_compounding_word_boundary(self) -> None:
        """Test that 'compounding' is detected only as a whole word."""
        # Should not match partial words like "noncompounding"
        pharmacy = PharmacyData(name="Noncompounding Pharmacy")
        result = rule_based_classify(pharmacy)
        
        # Should not be detected as compounding
        assert result.is_compounding is False


class TestIndependentPharmacyClassification:
    """Tests for independent pharmacy classification logic."""
    
    def test_default_independent(self) -> None:
        """Test default classification as independent pharmacy."""
        pharmacy = PharmacyData(name="Local Family Pharmacy")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as independent
        assert result.is_chain is False
        assert result.is_compounding is False
        assert result.confidence == 0.5  # Default confidence
        assert "No chain identifiers" in result.explanation
        assert result.source == ClassificationSource.RULE_BASED
    
    def test_empty_name(self) -> None:
        """Test classification with empty pharmacy name."""
        pharmacy = PharmacyData(name="")
        result = rule_based_classify(pharmacy)
        
        # Should be classified as independent with low confidence
        assert result.is_chain is False
        assert result.is_compounding is False
        assert result.confidence == 0.5
        assert "No chain identifiers" in result.explanation


class TestEdgeCases:
    """Tests for edge cases in rule-based classification."""
    
    def test_none_name(self) -> None:
        """Test classification with None as pharmacy name."""
        pharmacy = PharmacyData(name=None)
        result = rule_based_classify(pharmacy)
        
        # Should handle None gracefully
        assert result.is_chain is False
        assert result.is_compounding is False
    
    def test_with_special_characters(self) -> None:
        """Test classification with special characters in name."""
        pharmacy = PharmacyData(name="***CVS*** Pharmacy!!!")
        result = rule_based_classify(pharmacy)
        
        # Should still detect CVS despite special characters
        assert result.is_chain is True
        assert "CVS" in result.explanation
    
    def test_with_mixed_case(self) -> None:
        """Test classification with mixed case in name."""
        pharmacy = PharmacyData(name="cVs PhArMaCy")
        result = rule_based_classify(pharmacy)
        
        # Should be case-insensitive
        assert result.is_chain is True
        assert "CVS" in result.explanation


if __name__ == "__main__":
    pytest.main()
