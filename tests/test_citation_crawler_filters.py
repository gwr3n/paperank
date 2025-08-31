import unittest
from unittest import mock

class TestCrawlerFilters(unittest.TestCase):
    def test_crawl_filters_min_year(self):
        # Arrange: mock neighborhood expansion and Crossref metadata years
        import paperank.citation_crawler as cc

        def mock_get_citation_neighborhood(d, forward_steps=1, backward_steps=1, progress=False):
            # include root and two neighbors
            return [d, "x/1", "y/2"]

        def mock_get_work_metadata(doi, timeout=20):
            # Provide Crossref-like envelopes with different years
            if doi == "x/1":
                return {"message": {"issued": {"date-parts": [[2010]]}}}
            if doi == "y/2":
                return {"message": {"issued": {"date-parts": [[2020]]}}}
            return {"message": {}}

        with mock.patch.object(cc, "get_citation_neighborhood", side_effect=mock_get_citation_neighborhood), \
             mock.patch.object(cc, "get_work_metadata", side_effect=mock_get_work_metadata):
            # Act: require min_year=2015
            res = cc.crawl_citation_neighborhood(["root/0"], steps=1, min_year=2015, progress=False)

        # Assert: y/2 (2020) remains; x/1 (2010) filtered out
        self.assertIn("y/2", res)
        self.assertNotIn("x/1", res)

    def test_crawl_filters_min_citations(self):
        # Arrange: mock neighborhood expansion and citation counts
        import paperank.citation_crawler as cc

        def mock_get_citation_neighborhood(d, forward_steps=1, backward_steps=1, progress=False):
            # include root and three neighbors
            return [d, "x/1", "y/2", "z/3"]

        def oc_response(doi, timeout=20):
            # Simulate different citation counts from OpenCitations
            counts = {
                "x/1": 0,
                "y/2": 2,
                "z/3": 5,
            }
            c = counts.get(doi, 0)
            # get_citing_dois returns a dict with list "citing_dois"
            return {"article_doi": doi, "citing_dois": [f"c/{i}" for i in range(c)]}

        with mock.patch.object(cc, "get_citation_neighborhood", side_effect=mock_get_citation_neighborhood), \
             mock.patch.object(cc, "get_citing_dois", side_effect=oc_response):
            # Act: require at least 3 citations
            res = cc.crawl_citation_neighborhood(["root/0"], steps=1, min_citations=3, progress=False)

        # Assert: only z/3 survives the min_citations filter
        self.assertIn("z/3", res)
        self.assertNotIn("x/1", res)
        self.assertNotIn("y/2", res)

if __name__ == "__main__":
    unittest.main()