import pytest

from pharmacy_scraper.classification.classifier import (
    Classifier,
    clear_classification_cache,
    get_cache_stats,
)


def make_pharmacy(name="Acme Pharmacy", address="123 Main St"):
    return {"name": name, "address": address}


def test_cache_hit_and_stats():
    clear_classification_cache()
    clf = Classifier(client=None)  # force rule-based only

    # First call: miss -> store
    r1 = clf.classify_pharmacy(make_pharmacy())
    stats1 = get_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["stores"] == 1
    assert stats1["hits"] == 0

    # Second call: hit
    r2 = clf.classify_pharmacy(make_pharmacy())
    stats2 = get_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1
    assert r2.source.name == "CACHE"


def test_force_reclassification_bypasses_cache_and_overwrites():
    clear_classification_cache()
    clf = Classifier(client=None)

    # Seed cache
    _ = clf.classify_pharmacy(make_pharmacy())
    stats0 = get_cache_stats()
    assert stats0["stores"] == 1

    # Force reclass: should bypass cache and overwrite
    r_forced = clf.classify_pharmacy(make_pharmacy(), force_reclassification=True)
    stats1 = get_cache_stats()
    assert stats1["invalidations"] == 1
    assert stats1["stores"] == 2  # overwritten as new store
    assert r_forced.source.name in ("RULE_BASED", "PERPLEXITY")


def test_clear_cache_resets_stats():
    clear_classification_cache()
    clf = Classifier(client=None)
    _ = clf.classify_pharmacy(make_pharmacy())
    stats_before = get_cache_stats()
    assert stats_before["stores"] == 1

    clear_classification_cache()
    stats_after = get_cache_stats()
    assert stats_after == {"hits": 0, "misses": 0, "stores": 0, "invalidations": 0}
