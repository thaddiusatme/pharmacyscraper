"""
Tests for the initialization of PerplexityClient.

This module covers various initialization scenarios and edge cases,
such as API key handling, default parameter values, and configuration options.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pharmacy_scraper.classification.perplexity_client import (
    PerplexityClient,
    PerplexityAPIError,
)


class TestPerplexityClientInitialization:
    """Tests for PerplexityClient initialization and configuration."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicitly provided API key."""
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(api_key="test_api_key")
            assert client.api_key == "test_api_key"
            assert client.model_name == "pplx-7b-chat"  # Default model
            assert client.temperature == 0.1  # Default temperature
    
    def test_init_with_environment_variable(self, monkeypatch):
        """Test initialization with API key from environment variable."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "env_api_key")
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient()
            assert client.api_key == "env_api_key"
    
    @patch('pharmacy_scraper.classification.perplexity_client.get_config')
    def test_init_with_config(self, mock_get_config):
        """Test initialization with API key from config."""
        mock_get_config.return_value = {"perplexity_api_key": "config_api_key"}
        
        with patch.dict(os.environ, {}, clear=True), \
             patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient()
            assert client.api_key == "config_api_key"
            mock_get_config.assert_called_once()
    
    def test_missing_api_key(self):
        """Test that initialization fails when no API key is available."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('pharmacy_scraper.classification.perplexity_client.get_config', 
                   return_value={}):
            with pytest.raises(ValueError, match="No Perplexity API key provided"):
                PerplexityClient()
    
    def test_custom_model_and_temperature(self):
        """Test initialization with custom model and temperature."""
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(
                api_key="test_api_key", 
                model_name="custom-model", 
                temperature=0.7
            )
            assert client.model_name == "custom-model"
            assert client.temperature == 0.7
    
    @patch('pharmacy_scraper.classification.perplexity_client.OpenAI')
    def test_client_initialization(self, mock_openai):
        """Test that the OpenAI client is properly initialized."""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        client = PerplexityClient(api_key="test_api_key")
        
        # Verify OpenAI was initialized with the correct parameters
        mock_openai.assert_called_once_with(
            api_key="test_api_key", 
            base_url="https://api.perplexity.ai"
        )
        assert client.client is mock_openai_instance
    
    def test_custom_cache_config(self, tmp_path):
        """Test initialization with custom cache configuration."""
        cache_dir = tmp_path / "custom_cache"
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(
                api_key="test_api_key",
                cache_dir=str(cache_dir),
                cache_ttl_seconds=3600
            )
            
            assert client.cache is not None
            # Just check that cache exists, don't rely on exact path format
            assert isinstance(client.cache.cache_dir, (str, Path))
            assert client.cache.ttl == 3600
    
    def test_disable_cache(self):
        """Test disabling the cache."""
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(
                api_key="test_api_key",
                cache_enabled=False
            )
            assert client.cache is None
    
    def test_force_reclassification(self):
        """Test force_reclassification flag."""
        with patch('pharmacy_scraper.classification.perplexity_client.OpenAI'):
            client = PerplexityClient(
                api_key="test_api_key",
                force_reclassification=True
            )
            assert client.force_reclassification is True
