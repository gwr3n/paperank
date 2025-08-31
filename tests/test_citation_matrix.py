import unittest
from unittest.mock import patch

from paperank.citation_crawler import get_citation_neighborhood
from paperank.citation_matrix import build_citation_sparse_matrix


class TestCitationMatrix(unittest.TestCase):
    def setUp(self):
        self.test_doi = "10.1016/j.ejor.2016.12.001"
        self.forward_steps = 1
        self.backward_steps = 1

    def test_build_citation_sparse_matrix(self):
        doi_list = get_citation_neighborhood(
            self.test_doi, forward_steps=self.forward_steps, backward_steps=self.backward_steps, progress=False
        )
        matrix, doi_to_idx = build_citation_sparse_matrix(doi_list, max_workers=4, progress=True)
        self.assertIsNotNone(matrix)
        self.assertIsInstance(doi_to_idx, dict)
        self.assertEqual(matrix.shape[0], len(doi_to_idx))
        self.assertEqual(matrix.shape[1], len(doi_to_idx))

    @patch("paperank.citation_matrix.get_cited_dois", side_effect=Exception("network down"))
    @patch("paperank.citation_matrix.get_citing_dois", side_effect=Exception("network down"))
    def test_build_matrix_with_network_failures(self, *_):
        doi_list = [self.test_doi]
        matrix, mapping = build_citation_sparse_matrix(doi_list, max_workers=None, progress=True)
        self.assertEqual(matrix.shape, (1, 1))
        self.assertEqual(matrix.nnz, 0)
        self.assertIn(self.test_doi, mapping)


if __name__ == "__main__":
    unittest.main()
