import unittest
from paperank.citation_matrix import build_citation_sparse_matrix
from paperank.citation_crawler import get_citation_neighborhood

class TestCitationMatrix(unittest.TestCase):
    def setUp(self):
        self.test_doi = "10.1016/j.ejor.2016.12.001"
        self.forward_steps = 1
        self.backward_steps = 1

    def test_get_citation_neighborhood(self):
        doi_list = get_citation_neighborhood(
            self.test_doi,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            progress=False
        )
        self.assertIsInstance(doi_list, list)
        self.assertIn(self.test_doi, doi_list)

    def test_build_citation_sparse_matrix(self):
        doi_list = get_citation_neighborhood(
            self.test_doi,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            progress=False
        )
        matrix, doi_to_idx = build_citation_sparse_matrix(doi_list, max_workers=4, progress=True)
        self.assertIsNotNone(matrix)
        self.assertIsInstance(doi_to_idx, dict)
        self.assertEqual(matrix.shape[0], len(doi_to_idx))
        self.assertEqual(matrix.shape[1], len(doi_to_idx))

if __name__ == "__main__":
    unittest.main()