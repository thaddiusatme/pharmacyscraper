"""
Tests for the RateLimiter class in the Perplexity client.
"""
import time
import pytest
import threading
from unittest.mock import patch, MagicMock
from pharmacy_scraper.classification.perplexity_client import RateLimiter

class TestRateLimiter:
    """Test cases for the RateLimiter class."""
    
    def test_rate_limiter_with_zero_rate(self):
        """Test RateLimiter with zero rate limit (no delay)."""
        limiter = RateLimiter(0)  # No rate limiting
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be almost instantaneous
    
    def test_rate_limiter_with_positive_rate(self):
        """Test RateLimiter with a positive rate limit."""
        requests_per_minute = 60  # 1 request per second
        limiter = RateLimiter(requests_per_minute)
        
        # First call should not be delayed
        start_time = time.time()
        limiter.wait()
        first_call_time = time.time() - start_time
        assert first_call_time < 0.1  # Should be almost instantaneous
        
        # Second call should be delayed by approximately 1 second
        start_time = time.time()
        limiter.wait()
        second_call_time = time.time() - start_time
        assert 0.9 <= second_call_time <= 1.1  # Allow some tolerance for test execution
    
    def test_rate_limiter_with_high_rate(self):
        """Test RateLimiter with a high rate limit."""
        requests_per_minute = 600  # 10 requests per second
        limiter = RateLimiter(requests_per_minute)
        
        # First call should not be delayed
        start_time = time.time()
        limiter.wait()
        first_call_time = time.time() - start_time
        assert first_call_time < 0.1  # Should be almost instantaneous
        
        # Second call should be delayed by approximately 0.1 seconds
        start_time = time.time()
        limiter.wait()
        second_call_time = time.time() - start_time
        assert 0.05 <= second_call_time <= 0.15  # Allow some tolerance for test execution
    
    def test_rate_limiter_with_mocked_time(self, monkeypatch):
        """Test RateLimiter with mocked time to verify sleep behavior."""
        class MockTime:
            def __init__(self):
                self._time = 0.0
            
            def time(self):
                return self._time
                
            def sleep(self, seconds):
                self._time += seconds
        
        mock_time = MockTime()
        
        # Apply the monkey patches
        monkeypatch.setattr(time, 'time', mock_time.time)
        monkeypatch.setattr(time, 'sleep', mock_time.sleep)
        
        # Test with 2 requests per second (min_interval = 0.5s)
        requests_per_minute = 120
        limiter = RateLimiter(requests_per_minute)
        
        # First call - no delay expected
        start_time = mock_time.time()
        limiter.wait()
        elapsed = mock_time.time() - start_time
        assert elapsed == 0.0, "First call should not sleep"
        
        # Second call immediately after - should sleep for 0.5s
        start_time = mock_time.time()
        limiter.wait()
        elapsed = mock_time.time() - start_time
        assert elapsed == 0.5, f"Expected 0.5s sleep, got {elapsed}s"
        
        # Third call after waiting 0.25s - should sleep for 0.25s
        mock_time._time += 0.25
        start_time = mock_time.time()
        limiter.wait()
        elapsed = mock_time.time() - start_time
        assert abs(elapsed - 0.25) < 1e-9, f"Expected 0.25s sleep, got {elapsed}s"
    
    def test_rate_limiter_with_negative_rate(self):
        """Test RateLimiter with negative rate limit (should be treated as zero)."""
        limiter = RateLimiter(-1)  # Should be treated as 0 (no rate limiting)
        start_time = time.time()
        limiter.wait()
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be almost instantaneous
    
    def test_concurrent_access(self, monkeypatch):
        """Test RateLimiter with concurrent access from multiple threads."""
        # Use a lock to protect shared state
        from threading import Lock
        
        # Setup mock time and sleep
        mock_time = [0.0]  # Start at time 0.0
        time_lock = Lock()
        
        def mock_time_func():
            with time_lock:
                return mock_time[0]
                
        def mock_sleep(seconds):
            with time_lock:
                mock_time[0] += seconds
        
        # Apply the monkey patches
        monkeypatch.setattr(time, 'time', mock_time_func)
        monkeypatch.setattr(time, 'sleep', mock_sleep)
        
        # Setup rate limiter (1 request per second)
        limiter = RateLimiter(60)
        results = []
        result_lock = Lock()
        
        def worker():
            limiter.wait()
            with result_lock:
                results.append(mock_time_func())
        
        # Create and start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Let threads run
        for t in threads:
            t.join()
        
        # Verify that requests were rate-limited with at least 1s between them
        assert len(results) == 3
        assert results[1] - results[0] >= 0.99  # Should be at least 1s apart
        assert results[2] - results[1] >= 0.99  # Should be at least 1s apart
