import unittest

from paperank.citation_crawler import _cached_cited, _cached_citing
from paperank.citation_crawler import clear_caches as clear_crawler_caches
from paperank.crossref import clear_caches as clear_crossref_caches


class TestCaches(unittest.TestCase):
    def test_clear_crawler_caches(self):
        # Populate caches
        _ = _cached_cited("10.1234/foo")
        _ = _cached_citing("10.1234/bar")
        # Clear and ensure next call refills without error (content not asserted)
        clear_crawler_caches()
        _ = _cached_cited("10.1234/foo")
        _ = _cached_citing("10.1234/bar")

    def test_clear_crossref_cache(self):
        # Just ensure no error
        clear_crossref_caches()


if __name__ == "__main__":
    unittest.main()
