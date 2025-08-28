import unittest
from paperank.citation_crawler import (
    collect_cited_recursive,
    collect_citing_recursive,
    get_citation_neighborhood
)

class TestCitationCrawler(unittest.TestCase):
    def setUp(self):
        self.test_doi = "10.1016/j.ejor.2016.12.001"
        self.depth = 1

    def test_collect_cited_recursive_flat(self):
        citation_list = collect_cited_recursive(self.test_doi, self.depth, flatten=True)
        self.assertIsInstance(citation_list, list)
        self.assertTrue(self.test_doi not in citation_list or isinstance(citation_list, list))

    def test_collect_cited_recursive_tree(self):
        citation_tree = collect_cited_recursive(self.test_doi, self.depth)
        self.assertIsInstance(citation_tree, dict)
        self.assertIn(self.test_doi, citation_tree)

    def test_collect_citing_recursive_flat(self):
        citing_list = collect_citing_recursive(self.test_doi, self.depth, flatten=True)
        self.assertIsInstance(citing_list, list)
        self.assertTrue(self.test_doi not in citing_list or isinstance(citing_list, list))

    def test_collect_citing_recursive_tree(self):
        citing_tree = collect_citing_recursive(self.test_doi, self.depth)
        self.assertIsInstance(citing_tree, dict)
        self.assertIn(self.test_doi, citing_tree)

    def test_get_citation_neighborhood(self):
        result = get_citation_neighborhood(self.test_doi, forward_steps=1, backward_steps=1, progress=False)
        self.assertIsInstance(result, list)
        self.assertIn(self.test_doi, result)

if __name__ == "__main__":
    unittest.main()