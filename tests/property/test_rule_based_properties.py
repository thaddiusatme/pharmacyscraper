import os
import random
import string

from pharmacy_scraper.classification.classifier import rule_based_classify


def _rand_word(n: int = 8) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(n))


def _rand_pharmacy() -> dict:
    # Random minimal pharmacy dict with optional fields
    name = random.choice([
        f"{_rand_word()} Pharmacy",
        f"{_rand_word()} Compounding",
        f"{_rand_word()} Drugstore",
        _rand_word(10),
    ])
    address = random.choice([
        f"{random.randint(1, 9999)} {_rand_word()} St",
        f"{random.randint(1, 999)} {_rand_word()} Ave",
        "",
    ])
    return {"name": name, "address": address}


def test_rule_based_classify_is_deterministic_on_same_input():
    seed = int(os.getenv("PYTEST_SEED", "12345"))
    random.seed(seed)
    item = _rand_pharmacy()
    a = rule_based_classify(item)
    b = rule_based_classify(item)
    assert a.is_chain == b.is_chain
    assert a.is_compounding == b.is_compounding
    assert a.confidence == b.confidence
    assert type(a.explanation) is type(b.explanation)


def test_rule_based_classify_confidence_bounds_and_types():
    for _ in range(25):
        res = rule_based_classify(_rand_pharmacy())
        assert 0.0 <= float(res.confidence) <= 1.0
        assert isinstance(res.is_chain, bool)
        assert isinstance(res.is_compounding, bool)
        assert isinstance(res.explanation, str)


def test_rule_based_compounding_keyword_bias():
    # Ensure obvious compounding hint increases likelihood
    obvious = {"name": "Acme Compounding", "address": "123 Main"}
    neutral = {"name": "Acme Pharmacy", "address": "123 Main"}
    a = rule_based_classify(obvious)
    b = rule_based_classify(neutral)
    # At least ensure obvious is not strictly less confident than neutral
    assert a.confidence >= b.confidence
