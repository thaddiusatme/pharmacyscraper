"""Unit tests for Google Places verification helpers.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.verification import google_places as gp
from src.utils.api_usage_tracker import CreditLimitExceededError


_SAMPLE_PHARMACY = {
    "name": "Test Pharmacy",
    "latitude": 37.0001,
    "longitude": -122.0001,
    "address": "123 Main St",
    "city": "Test City",
    "state": "CA",
}


def _make_mock_tracker(budget_ok: bool = True):
    """Return a MagicMock implementing the credit_tracker interface."""
    tracker = MagicMock()
    tracker.check_credit_available.return_value = budget_ok
    # record_usage is called but its return value is not used.
    tracker.record_usage = MagicMock()
    return tracker


@pytest.fixture()
def mock_tracker_ok():
    with patch.object(gp, "credit_tracker", _make_mock_tracker(True)):
        yield


@pytest.fixture()
def mock_tracker_exhausted():
    with patch.object(gp, "credit_tracker", _make_mock_tracker(False)):
        yield


def _make_gmaps(return_dict=None, *, raise_exc: bool = False):
    client = MagicMock()
    if raise_exc:
        client.places.side_effect = Exception("api fail")
    else:
        client.places.return_value = return_dict or {}
    return client


def test_verify_success_high_confidence(mock_tracker_ok):
    """Happy-path: similar name and close distance -> verified True."""
    gmaps = _make_gmaps(
        {
            "results": [
                {
                    "place_id": "abc123",
                    "name": "Test Pharmacy",
                    "geometry": {"location": {"lat": 37.0002, "lng": -122.0002}},
                }
            ]
        }
    )

    result = gp.verify_pharmacy(dict(_SAMPLE_PHARMACY), gmaps=gmaps)
    assert result["verified"] is True
    assert result["google_place_id"] == "abc123"
    # confidence should be reasonably high (>0.6)
    assert result["verification_confidence"] >= 0.6


def test_verify_low_confidence(mock_tracker_ok):
    """Different name and far distance -> verified False."""
    gmaps = _make_gmaps(
        {
            "results": [
                {
                    "place_id": "zzz",
                    "name": "Completely Different Name",
                    "geometry": {"location": {"lat": 38.0, "lng": -123.0}},
                }
            ]
        }
    )

    result = gp.verify_pharmacy(dict(_SAMPLE_PHARMACY), gmaps=gmaps)
    assert result["verified"] is False
    assert result["verification_confidence"] < 0.6


def test_verify_api_error(mock_tracker_ok):
    """If Google Maps client errors, verify_pharmacy should fail soft."""
    gmaps = _make_gmaps(raise_exc=True)
    res = gp.verify_pharmacy(dict(_SAMPLE_PHARMACY), gmaps=gmaps)
    assert res["verified"] is False
    assert res["google_place_id"] is None
    assert res["verification_confidence"] == 0.0


def test_budget_exhausted_raises(mock_tracker_exhausted):
    """When budget is exhausted, verify_pharmacy should raise an explicit error."""
    gmaps = _make_gmaps({"results": []})
    with pytest.raises(CreditLimitExceededError):
        gp.verify_pharmacy(dict(_SAMPLE_PHARMACY), gmaps=gmaps)
