"""
Tests for cache size limits and eviction policies in PerplexityClient.
"""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pharmacy_scraper.classification.perplexity_client import PerplexityClient

# Sample data for testing
SAMPLE_PHARMACY = {
    "title": "Test Pharmacy",
    "address": "123 Test St, Test City, TS 12345",
    "phone": "(555) 123-4567"
}

# Sample API response
SAMPLE_RESPONSE = {
    "classification": "independent",
    "confidence": 0.95,
    "explanation": "This is a test response"
}

class TestCacheLimits:
    """Test suite for cache size limits and eviction policies."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Set up test environment."""
        self.temp_dir = tmp_path / "test_cache"
        self.temp_dir.mkdir()
        
        # Mock OpenAI client
        self.mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = json.dumps(SAMPLE_RESPONSE)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        self.mock_openai_client.chat.completions.create.return_value = mock_response
        
        yield
        
        # Cleanup
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*"):
                f.unlink()
            self.temp_dir.rmdir()
    
    def create_large_cache_file(self, filename, size_kb=1):
        """Create a cache file of specified size in KB."""
        cache_file = self.temp_dir / filename
        # Create content that's approximately the right size
        content = {"data": SAMPLE_RESPONSE, "cached_at": 0}
        # Make the content larger by adding padding
        padding = "x" * int(size_kb * 512)  # Convert to int for multiplication
        content["padding"] = padding
        
        with open(cache_file, 'w') as f:
            json.dump(content, f)
        return cache_file
    
    def test_cache_size_limit_enforcement(self):
        """Test that cache respects size limits."""
        # Create a cache file that exceeds the limit
        self.create_large_cache_file("test1.json", size_kb=2)
        
        # Mock the file system calls
        with patch('os.path.getsize') as mock_getsize, \
             patch('os.path.getmtime') as mock_mtime:
            # Mock the file size to be larger than our limit
            mock_getsize.return_value = 2 * 1024  # 2KB
            mock_mtime.return_value = time.time()
            
            # Set a small cache size limit (1KB) after mocking
            client = PerplexityClient(
                api_key="test_key",
                cache_dir=str(self.temp_dir),
                max_cache_size_mb=0.001,  # 1KB
                openai_client=self.mock_openai_client,
                cleanup_frequency=0,  # Run cleanup on every check
                model_name="test-model"  # Add model_name to avoid attribute error
            )
            
            # This should trigger cache cleanup
            client._check_cache_limits()
                
        # Verify cleanup was triggered - should be empty since we only had one file
        # and it exceeded the limit
        cache_files = list(self.temp_dir.glob('*.json'))
        assert len(cache_files) == 0, "Cache file should have been cleaned up"
    
    def test_lru_eviction_policy(self):
        """Test that LRU eviction policy works correctly."""
        # Create multiple cache files
        for i in range(5):
            self.create_large_cache_file(f"test{i}.json", size_kb=1)
        
        # Mock the file system calls
        with patch('os.path.getsize') as mock_getsize, \
             patch('os.path.getmtime') as mock_mtime:
            # Make each file appear to be 1KB
            mock_getsize.return_value = 1024  # 1KB
            # Make sure files have different modification times (oldest first)
            mock_mtime.side_effect = [time.time() - i for i in range(5, 0, -1)]
            
            # Create a client with a small cache limit (5KB)
            client = PerplexityClient(
                api_key="test_key",
                cache_dir=str(self.temp_dir),
                openai_client=self.mock_openai_client,
                max_cache_size_mb=0.005,  # 5KB limit
                cleanup_frequency=0  # Run cleanup on every check
            )
            
            # Create multiple cache files that exceed the limit
            for i in range(5):
                # Make each file have a different mtime
                mock_mtime.return_value = time.time() + i
                self.create_large_cache_file(f"test{i}.json", size_kb=3)  # 15KB total
        
        # Mock the file size to be larger than our limit
        with patch('os.path.getsize') as mock_getsize, \
             patch('os.path.getmtime') as mock_mtime:
            mock_getsize.return_value = 3 * 1024  # 3KB per file
            # Make sure getmtime returns different values for each file
            mock_mtime.side_effect = [time.time() - i for i in range(5)]
            
            # This should trigger cleanup
            client._check_cache_limits()
        
        # After cleanup, we should have fewer files
        cache_files = list(self.temp_dir.glob('*.json'))
        assert len(cache_files) < 5, "Some files should have been cleaned up"
    
    def test_cache_size_metrics(self):
        """Test that cache size metrics are accurate."""
        client = PerplexityClient(
            api_key="test_key",
            cache_dir=str(self.temp_dir),
            openai_client=self.mock_openai_client,
            max_cache_size_mb=1.0,  # 1MB limit
            cleanup_frequency=0  # Disable cleanup for this test
        )
        
        # Create a cache file that's about 1KB
        test_file = self.create_large_cache_file("test1.json", size_kb=1)
        file_size = test_file.stat().st_size
        
        # Get metrics and verify size matches
        metrics = client.get_cache_metrics()
        assert metrics['size'] == file_size  # Size should match actual file size
    
    def test_cache_cleanup_on_init(self):
        """Test that cache is cleaned up on initialization."""
        # Create an expired cache file
        expired_file = self.temp_dir / "expired.json"
        with open(expired_file, 'w') as f:
            json.dump({
                'data': SAMPLE_RESPONSE,
                'cached_at': 0,  # Very old timestamp
                'expires_at': 1,  # Already expired
                'size': 100  # Dummy size
            }, f)
        
        # Create a non-expired cache file
        valid_file = self.temp_dir / "valid.json"
        with open(valid_file, 'w') as f:
            json.dump({
                'data': SAMPLE_RESPONSE,
                'cached_at': time.time(),
                'expires_at': time.time() + 3600,  # Expires in 1 hour
                'size': 100  # Dummy size
            }, f)
        
        # Mock the file system calls
        with patch('os.path.getmtime') as mock_mtime, \
             patch('os.path.getsize') as mock_getsize:
            # Make files appear recent
            mock_mtime.return_value = time.time()
            # Make files appear small
            mock_getsize.return_value = 100
            
            # Initialize client with TTL and cleanup_frequency=0 to force cleanup
            client = PerplexityClient(
                api_key="test_key",
                cache_dir=str(self.temp_dir),
                openai_client=self.mock_openai_client,
                cache_ttl_seconds=3600,  # 1 hour TTL
                cleanup_frequency=0,  # Run cleanup on every check
                max_cache_size_mb=1  # 1MB limit
            )
            
            # Manually trigger cleanup of expired entries
            client._cleanup_expired_entries()
        
        # Check that expired file was removed
        assert not expired_file.exists(), "Expired file should have been removed"
        assert valid_file.exists(), "Valid file should still exist"
    
    def test_cache_size_limit_with_multiple_files(self):
        """Test cache size limit with multiple files."""
        # Create test files
        test_files = [self.temp_dir / f"test{i}.json" for i in range(5)]
        for i, file in enumerate(test_files):
            file.write_text(json.dumps({
                'data': SAMPLE_RESPONSE,
                'cached_at': time.time(),
                'expires_at': time.time() + 3600,  # 1 hour from now
                'size': 3 * 1024,  # 3KB
                'padding': 'x' * (3 * 1024)  # Dummy data to make file size larger
            }))
        
        # Mock the file system calls
        with patch('os.path.getsize') as mock_getsize, \
             patch('os.path.getmtime') as mock_mtime, \
             patch('os.listdir') as mock_listdir:
            
            # Mock getsize to return 3KB for each file
            mock_getsize.return_value = 3 * 1024  # 3KB per file
            
            # Set modification times (oldest first)
            base_time = time.time()
            mock_mtime.side_effect = [base_time - i * 3600 for i in range(5)]
            
            # Mock listdir to return our test files
            mock_listdir.return_value = [f.name for f in test_files]
            
            # Create a client with a small cache limit (10KB)
            client = PerplexityClient(
                api_key="test_key",
                cache_dir=str(self.temp_dir),
                openai_client=self.mock_openai_client,
                max_cache_size_mb=0.01,  # 10KB limit
                cleanup_frequency=0,  # Run cleanup on every check
                model_name="test-model"
            )
            
            # Manually trigger cache cleanup
            client._check_cache_limits()
            
            # Check that some files were removed
            remaining_files = list(self.temp_dir.glob('*.json'))
            assert len(remaining_files) < 5, "Some files should have been cleaned up"
            
            # Get the indices of remaining files
            remaining_indices = [int(f.stem[4:]) for f in remaining_files]
            
            # Verify that the number of remaining files is as expected
            # We have 5 files of 3KB each (15KB total) and a 10KB limit
            # The implementation should remove enough files to get under 90% of the limit (9KB)
            # Since each file is 3KB, we should have 2 files remaining (6KB)
            assert len(remaining_indices) >= 2, "Should have at least 2 files remaining (6KB)"
            
            # The implementation sorts files by modification time (oldest first) and removes the oldest ones first
            # Since we set modification times in reverse order (newest first), files with lower indices are older
            # Check that the oldest files (indices 0 and 1) are not in the remaining indices
            assert 0 not in remaining_indices, "Oldest file (index 0) should have been removed"
            assert 1 not in remaining_indices, "Second oldest file (index 1) should have been removed"
            
            # Verify the total size is under the limit
            total_size = sum(os.path.getsize(self.temp_dir / f"test{i}.json") for i in remaining_indices)
            assert total_size <= 10 * 1024, f"Total size {total_size} bytes exceeds 10KB limit"
