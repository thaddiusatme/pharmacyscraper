import pytest

from pharmacy_scraper.plugins.validation import (
    validate_classifier_plugin,
    validate_data_source_plugin,
)
from pharmacy_scraper.plugins.interfaces import (
    DataSourcePlugin,
    BaseClassifierPlugin,
)


class GoodSource(DataSourcePlugin):
    name = "good_source"

    def fetch(self, query: dict, cfg: dict) -> list:
        return []


class BadSource:
    pass


class GoodClassifier(BaseClassifierPlugin):
    name = "good_classifier"

    def classify(self, item: dict, cfg: dict) -> dict:
        return {"label": "independent", "score": 0.9}


class BadClassifier(BaseClassifierPlugin):
    # missing name
    def classify(self, item: dict, cfg: dict) -> dict:
        return {"label": "independent", "score": 0.9}


def test_validate_classifier_plugin_accepts_good():
    validate_classifier_plugin(GoodClassifier)


def test_validate_classifier_plugin_rejects_missing_name():
    with pytest.raises(ValueError):
        validate_classifier_plugin(BadClassifier)


def test_validate_data_source_plugin_accepts_good():
    validate_data_source_plugin(GoodSource)


def test_validate_data_source_plugin_rejects_bad_type():
    with pytest.raises(ValueError):
        validate_data_source_plugin(BadSource)
