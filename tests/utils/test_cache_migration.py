import json
from pathlib import Path

from pharmacy_scraper.utils.cache_migration import migrate_cache_keys


def test_migrate_cache_keys_creates_typed_copies(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create two legacy cache files and one already-typed file
    (cache_dir / "foo_bar.json").write_text(json.dumps([{"name": "A"}]))
    (cache_dir / "baz.json").write_text(json.dumps([{"name": "B"}]))
    (cache_dir / "pharmacy:foo_bar.json").write_text(json.dumps([{"name": "A-typed"}]))

    stats = migrate_cache_keys(cache_dir, "vet_clinic", dry_run=False)

    # Should have created typed copies for both legacy files
    typed1 = cache_dir / "vet_clinic:foo_bar.json"
    typed2 = cache_dir / "vet_clinic:baz.json"
    assert typed1.exists(), "Expected typed copy for foo_bar.json"
    assert typed2.exists(), "Expected typed copy for baz.json"

    # Contents should be valid JSON
    data1 = json.loads(typed1.read_text())
    data2 = json.loads(typed2.read_text())
    assert isinstance(data1, list)
    assert isinstance(data2, list)

    # Already-typed pharmacy file remains untouched; not counted in migration
    assert (cache_dir / "pharmacy:foo_bar.json").exists()
    assert stats.migrated >= 2
