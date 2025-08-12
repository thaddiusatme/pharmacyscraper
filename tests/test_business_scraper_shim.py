import importlib
import sys


def test_top_level_alias_imports():
    # Import new namespace first
    bs = importlib.import_module('business_scraper')
    # Import old namespace
    ps = importlib.import_module('pharmacy_scraper')

    # Both entries should exist in sys.modules and be the same object
    assert sys.modules['business_scraper'] is sys.modules['pharmacy_scraper']
    assert bs is ps


def test_submodule_alias_utils_logger():
    # Choose a lightweight submodule that should not pull heavy third-party deps
    bs_logger = importlib.import_module('business_scraper.utils.logger')
    ps_logger = importlib.import_module('pharmacy_scraper.utils.logger')

    # They should resolve to the same module object
    assert bs_logger is ps_logger

    # Basic sanity: module has expected attributes
    assert hasattr(bs_logger, '__name__')
    # Access attributes that exist in the logger module
    assert hasattr(bs_logger, 'setup_logger')
    assert hasattr(bs_logger, 'get_console_logger')
