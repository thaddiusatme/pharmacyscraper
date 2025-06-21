"""Google Places verification utilities for Phase 2b of the pipeline.

This module checks pharmacy records returned by Apify/Classification against
Google Places to increase confidence that the location is an *independent* 
pharmacy with a valid mailing address.

Design goals
------------
• **Stateless** aside from optional file-system caching handled upstream.
• **Budget-aware** – every request consults ``credit_tracker`` before making an
  external call and records usage afterwards.
• **Fail-soft** – network / quota errors mark the pharmacy as *unverified* but
  never raise, so the pipeline keeps progressing.
"""
from __future__ import annotations

import os
import logging
import math
from difflib import SequenceMatcher
from typing import Dict, List

import googlemaps  # type: ignore

# Internal utilities
from src.utils.api_usage_tracker import credit_tracker, CreditLimitExceededError

logger = logging.getLogger(__name__)

# Approx. cost per Places *detail* request in USD (adjust as needed)
_COST_PER_LOOKUP = 0.02

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two GPS points."""
    R = 6371e3  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _similar(a: str, b: str) -> float:
    """Return [0,1] similarity score between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def _create_client() -> googlemaps.Client:
    """Create a ``googlemaps.Client`` using **GOOGLE_API_KEY** env var."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY or GOOGLE_MAPS_API_KEY environment variable not set – cannot run verification phase.")
    return googlemaps.Client(key=api_key)


def verify_pharmacy(pharmacy: Dict, gmaps: googlemaps.Client | None = None) -> Dict:
    """Verify a single pharmacy via Google Places.

    Parameters
    ----------
    pharmacy: dict
        Must contain at least ``name`` and either ``formatted_address`` *or*
        separate ``address``/``city``/``state`` keys.
    gmaps: googlemaps.Client | None
        Re-use an existing client for efficiency.

    Returns
    -------
    dict
        Updates containing keys:
        ``verified`` (bool), ``verification_confidence`` (float [0-1]),
        ``google_place_id`` (str | None)
    """

    if gmaps is None:
        gmaps = _create_client()

    # Budget guard
    if not credit_tracker.check_credit_available(_COST_PER_LOOKUP):
        raise CreditLimitExceededError("Budget exhausted before Google Places verification call")

    name: str = pharmacy.get("name", pharmacy.get("title", ""))
    address_parts = [
        pharmacy.get("formatted_address", pharmacy.get("address")),
        pharmacy.get("city"),
        pharmacy.get("state"),
    ]
    query = ", ".join([p for p in address_parts if p]) or name

    try:
        resp = gmaps.places(query=query, type="pharmacy")
        candidates = resp.get("results", [])
        if not candidates:
            # record usage even if empty – we did perform a call
            credit_tracker.record_usage(_COST_PER_LOOKUP, operation="google_places_zero")
            return {
                "verified": False,
                "verification_confidence": 0.0,
                "google_place_id": None,
            }

        best = candidates[0]
        place_id = best.get("place_id")
        gm_name = best.get("name", "")
        gm_location = best.get("geometry", {}).get("location", {})

        # Compute simple confidence metrics
        name_score = _similar(name, gm_name)
        distance_m = None
        
        # Handle location data properly - check for location dict or separate lat/lng fields
        pharmacy_location = pharmacy.get("location", {})
        if isinstance(pharmacy_location, str):
            # Handle string format like "{'lat': 33.5524189, 'lng': -112.0584158}"
            try:
                import ast
                pharmacy_location = ast.literal_eval(pharmacy_location)
            except:
                pharmacy_location = {}
        
        pharmacy_lat = pharmacy_location.get("lat") or pharmacy.get("latitude")
        pharmacy_lng = pharmacy_location.get("lng") or pharmacy.get("longitude")
        
        if ("lat" in gm_location and "lng" in gm_location and 
            pharmacy_lat is not None and pharmacy_lng is not None):
            try:
                distance_m = _haversine_distance(
                    float(pharmacy_lat), float(pharmacy_lng), 
                    gm_location["lat"], gm_location["lng"]
                )
            except (ValueError, TypeError):
                distance_m = None

        distance_score = 1.0 if distance_m is None else max(0.0, 1.0 - (distance_m / 500))  # 500 m threshold
        confidence = 0.7 * name_score + 0.3 * distance_score
        verified = confidence >= 0.6

        # Record API usage
        credit_tracker.record_usage(_COST_PER_LOOKUP, operation="google_places")

        return {
            "verified": verified,
            "verification_confidence": round(confidence, 3),
            "google_place_id": place_id,
        }

    except Exception as exc:  # noqa: BLE001 – capture any network / quota errors
        logger.error("Google Places verification failed for %s: %s", name, exc)
        # Still count cost because request likely went out or will on retry
        credit_tracker.record_usage(_COST_PER_LOOKUP, operation="google_places_error")
        return {
            "verified": False,
            "verification_confidence": 0.0,
            "google_place_id": None,
        }


def verify_batch(pharmacies: List[Dict], cache_dir: str | None = None) -> List[Dict]:
    """Batch-verify list of pharmacies. Currently stateless; ``cache_dir`` reserved for future."""
    gmaps = _create_client()
    out: List[Dict] = []
    for p in pharmacies:
        res = verify_pharmacy(p, gmaps=gmaps)
        p.update(res)
        out.append(p)
    return out
