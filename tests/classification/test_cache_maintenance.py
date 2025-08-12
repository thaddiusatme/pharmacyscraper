import types

import pytest

from pharmacy_scraper.classification.classifier import (
    Classifier,
    clear_classification_cache,
    get_cache_stats,
    invalidate_cache_key,
    prune_cache,
    purge_cache_older_than,
)
from pharmacy_scraper.utils.cache_keys import pharmacy_cache_key


def make_pharmacy(i: int):
    return {"name": f"Pharmacy {i}", "address": f"{i} Main St"}


def seed_entries(n: int):
    clf = Classifier(client=None)
    for i in range(n):
        clf.classify_pharmacy(make_pharmacy(i), use_llm=False)
    return clf


def test_invalidate_cache_key_removes_and_counts():
    clear_classification_cache()
    clf = seed_entries(2)

    key = pharmacy_cache_key(make_pharmacy(0), use_llm=False)
    stats_before = get_cache_stats()
    assert stats_before["stores"] == 2

    removed = invalidate_cache_key(key)
    assert removed is True

    # Removing again returns False, no double count
    removed2 = invalidate_cache_key(key)
    assert removed2 is False

    stats_after = get_cache_stats()
    assert stats_after["invalidations"] == stats_before["invalidations"] + 1


def test_prune_cache_keeps_max_entries():
    clear_classification_cache()
    _ = seed_entries(5)

    removed = prune_cache(3)
    assert removed == 2

    # Pruning again to same size should remove 0
    removed_again = prune_cache(3)
    assert removed_again == 0


def test_purge_cache_older_than(monkeypatch):
    clear_classification_cache()

    # Monkeypatch time to control ages
    times = [1000.0, 1001.0, 1002.0, 1003.0]
    idx = {"i": 0}

    def fake_time():
        return times[min(idx["i"], len(times) - 1)]

    # Patch time.time used inside module
    import pharmacy_scraper.classification.classifier as mod

    monkeypatch.setattr(mod.time, "time", fake_time)

    # Insert 3 entries at t=1000,1001,1002
    for _ in range(3):
        _ = Classifier(client=None).classify_pharmacy(make_pharmacy(idx["i"]), use_llm=False)
        idx["i"] += 1

    # Now current time t=1003
    idx["i"] = 3

    # Purge entries older than or equal to 2 seconds (t<=1001)
    removed = purge_cache_older_than(2.0)
    assert removed == 2

    # Purge with zero threshold should remove remaining (since 1003-1002>=0)
    removed2 = purge_cache_older_than(0.0)
    assert removed2 >= 1
