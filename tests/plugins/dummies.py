from pharmacy_scraper.plugins.interfaces import DataSourcePlugin, ClassifierPlugin


class DummySource(DataSourcePlugin):
    name = "dummy_source"

    def fetch(self, query: dict, cfg: dict) -> list:
        return [query]


class DummyClassifier(ClassifierPlugin):
    name = "dummy_classifier"

    def classify(self, item: dict, cfg: dict) -> dict:
        return {"label": "ok", "score": 1.0, "item": item}
