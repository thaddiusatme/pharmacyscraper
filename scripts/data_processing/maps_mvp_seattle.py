#!/usr/bin/env python3
"""
MVP: Fetch ~50 specialty/independent grocery locations near Seattle, WA using Apify Google Maps Scraper
and export a reviewable CSV for client outreach.

Usage:
  APIFY_TOKEN=xxxx python scripts/data_processing/maps_mvp_seattle.py --limit 50 --out output/locations/seattle_mvp.csv

Notes:
- Tries manifest-default actor ID first (nwua9Gu5YrADL7ZDj), falls back to 'apify/google-maps-scraper'.
- Dedupes by (lower(name), formatted_address) to keep unique places.
- Outputs columns: name, formatted_address, city, state, postal_code, lat, lng, phone, website, url, categories
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

try:
    from apify_client import ApifyClient
except Exception as e:
    print("Error: apify-client is not installed. Install with: pip install apify-client", file=sys.stderr)
    raise

DEFAULT_TERMS = [
    "specialty grocery",
    "independent grocery",
    "Mediterranean market",
    "gourmet market",
    "natural foods",
]

MANIFEST_DEFAULT_ACTOR_ID = os.getenv("APIFY_MAPS_ACTOR_ID", "nwua9Gu5YrADL7ZDj")
FALLBACK_ACTOR_ID = "apify/google-maps-scraper"


def build_search_strings(location: str, terms: List[str]) -> List[str]:
    return [f"{t} in {location}" for t in terms]


def start_run(client: ApifyClient, actor_id: str, search_strings: List[str], max_items: int) -> Dict[str, Any]:
    run_input = {
        "searchStringsArray": search_strings,
        "searchMatching": "all",
        "maxCrawledPlaces": max_items,
        "countryCode": "us",
        "language": "en",
        # keep outputs lean/cheap
        "maxImages": 0,
        "maxReviews": 0,
        "includeGeolocation": True,
        "includeOpeningHours": False,
        "includePeopleAlsoSearch": False,
        "includePopularTimes": False,
        "includeReviews": False,
        "includeReviewsSummary": False,
        "includeImages": False,
        "includeWebResults": False,
        "includeDetailUrl": True,
        "includePosition": True,
        "maxConcurrency": 5,
        "maxRequestRetries": 2,
        "requestTimeoutSecs": 60,
    }
    # Fire and poll
    run = client.actor(actor_id).call(run_input=run_input, wait_secs=0)
    return run


def wait_for_run(client: ApifyClient, run_id: str, max_wait_secs: int = 1800, poll_secs: int = 5) -> Dict[str, Any]:
    start = time.time()
    while True:
        run = client.run(run_id).get()
        status = run.get("status")
        if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):
            return run
        if time.time() - start > max_wait_secs:
            raise TimeoutError(f"Run {run_id} exceeded max wait {max_wait_secs}s")
        time.sleep(poll_secs)


def fetch_dataset_items(client: ApifyClient, dataset_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    ds = client.dataset(dataset_id)
    try:
        resp = ds.list_items()
        items = list(resp.items)
    except Exception:
        # fallback
        items = list(ds.iterate_items())
    return items


def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
    # Apify Google Maps outputs vary; try common fields
    name = it.get("name") or it.get("title")
    formatted_address = it.get("formattedAddress") or it.get("address") or ""
    location = it.get("location") or {}
    lat = location.get("lat") or location.get("latitude")
    lng = location.get("lng") or location.get("longitude")
    phone = it.get("phone") or it.get("phoneNumber")
    website = it.get("website") or it.get("domain")
    url = it.get("url") or it.get("link") or it.get("detailUrl")
    cats = it.get("categories") or it.get("categoryName")
    if isinstance(cats, list):
        categories = ", ".join([str(c) for c in cats])
    else:
        categories = str(cats) if cats else ""

    # Best-effort split for city/state/postal from formatted address
    city = state = postal = ""
    try:
        parts = [p.strip() for p in formatted_address.split(",")]
        if len(parts) >= 3:
            city = parts[-3]
            state_zip = parts[-2].split()
            if len(state_zip) >= 1:
                state = state_zip[0]
            if len(state_zip) >= 2:
                postal = state_zip[1]
    except Exception:
        pass

    return {
        "name": name,
        "formatted_address": formatted_address,
        "city": city,
        "state": state,
        "postal_code": postal,
        "lat": lat,
        "lng": lng,
        "phone": phone,
        "website": website,
        "url": url,
        "categories": categories,
        "source": "apify_google_maps",
    }


def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = ((it.get("name") or "").strip().lower(), (it.get("formatted_address") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main():
    parser = argparse.ArgumentParser(description="Seattle MVP discovery via Apify Maps")
    parser.add_argument("--location", default="Seattle, WA")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default="output/locations/seattle_mvp.csv")
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS)
    args = parser.parse_args()

    token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")
    if not token:
        print("Error: set APIFY_TOKEN or APIFY_API_TOKEN", file=sys.stderr)
        sys.exit(1)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    client = ApifyClient(token)

    # Determine actor to use
    actor_id = MANIFEST_DEFAULT_ACTOR_ID
    try:
        client.actor(actor_id).get()
    except Exception:
        print(f"Actor {actor_id} not accessible; falling back to {FALLBACK_ACTOR_ID}")
        actor_id = FALLBACK_ACTOR_ID

    search_strings = build_search_strings(args.location, args.terms)
    # Start and wait
    run = start_run(client, actor_id, search_strings, args.limit)
    run_id = run["id"] if isinstance(run, dict) else str(run)
    final = wait_for_run(client, run_id, max_wait_secs=1800, poll_secs=5)
    if final.get("status") != "SUCCEEDED":
        print(f"Run ended with status {final.get('status')}", file=sys.stderr)
        sys.exit(2)

    dataset_id = final.get("defaultDatasetId")
    if not dataset_id:
        print("No dataset id returned", file=sys.stderr)
        sys.exit(3)

    raw_items = fetch_dataset_items(client, dataset_id)
    norm = [normalize_item(it) for it in raw_items]
    deduped = dedupe(norm)

    # Trim to limit
    deduped = deduped[: args.limit]

    # Save CSV
    try:
        import pandas as pd
    except Exception:
        print("pandas not installed. Install with: pip install pandas", file=sys.stderr)
        sys.exit(4)

    import pandas as pd  # noqa
    df = pd.DataFrame(deduped)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(deduped)} locations to {args.out}")


if __name__ == "__main__":
    main()
