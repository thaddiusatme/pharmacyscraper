import pytest
from pharmacy_scraper.classification import classify_pharmacy

def test_classify_pharmacy_type_error():
    with pytest.raises(TypeError):
        classify_pharmacy({}, force_reclassification=True)
