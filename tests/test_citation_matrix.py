import unittest
from unittest.mock import patch

from paperank.citation_crawler import get_citation_neighborhood
from paperank.citation_matrix import build_citation_sparse_matrix, _cached_cited, _cached_citing


class TestCitationMatrix(unittest.TestCase):
    def setUp(self):
        self.test_doi = "10.1016/j.ejor.2016.12.001"
        self.forward_steps = 1
        self.backward_steps = 1
        # Clear caches to avoid cross-test contamination
        _cached_cited.cache_clear()
        _cached_citing.cache_clear()

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
    
    @patch("paperank.citation_matrix.get_cited_dois")
    def test_cached_cited_returns_tuple_and_uses_cache(self, mock_get_cited):
        mock_get_cited.return_value = {"cited_dois": ["10.1000/a", "10.1000/b"]}
        doi = "10.5555/cache.test"

        out1 = _cached_cited(doi)
        out2 = _cached_cited(doi)

        self.assertEqual(out1, ("10.1000/a", "10.1000/b"))
        self.assertEqual(out1, out2)
        mock_get_cited.assert_called_once_with(doi)

    @patch("paperank.citation_matrix.get_citing_dois")
    def test_cached_citing_returns_tuple_and_uses_cache(self, mock_get_citing):
        mock_get_citing.return_value = {"citing_dois": ["10.2000/x", "10.2000/y"]}
        doi = "10.5555/cache.citing"

        out1 = _cached_citing(doi)
        out2 = _cached_citing(doi)

        self.assertEqual(out1, ("10.2000/x", "10.2000/y"))
        self.assertEqual(out1, out2)
        mock_get_citing.assert_called_once_with(doi)

    @patch("paperank.citation_matrix.get_cited_dois")
    def test_cached_cited_handles_missing_or_none_values(self, mock_get_cited):
        doi = "10.5555/none"

        mock_get_cited.return_value = {"cited_dois": None}
        self.assertEqual(_cached_cited(doi), tuple())

        _cached_cited.cache_clear()
        mock_get_cited.return_value = {}
        self.assertEqual(_cached_cited(doi), tuple())

    @patch("paperank.citation_matrix.get_citing_dois", side_effect=Exception("boom"))
    def test_cached_citing_handles_exceptions_as_empty_tuple(self, _):
        doi = "10.5555/boom"
        self.assertEqual(_cached_citing(doi), tuple())

    @patch("paperank.citation_matrix.get_citing_dois")
    @patch("paperank.citation_matrix.get_cited_dois")
    def test_include_citing_adds_edge_from_citer_to_target(self, mock_get_cited, mock_get_citing):
        # Arrange: B cites A; both A and B are in the list
        doi_a = "10.1000/a"
        doi_b = "10.1000/b"

        def citing_side_effect(doi):
            if doi == doi_a:
                return {"citing_dois": [doi_b]}
            if doi == doi_b:
                return {"citing_dois": []}
            return {"citing_dois": []}

        mock_get_citing.side_effect = citing_side_effect
        mock_get_cited.return_value = {"cited_dois": []}

        # Act
        matrix, mapping = build_citation_sparse_matrix(
            [doi_a, doi_b], include_citing=True, max_workers=None, progress=False
        )

        # Assert: Edge should be from B -> A (row=B, col=A)
        idx_a = mapping[doi_a]
        idx_b = mapping[doi_b]
        rows, cols = matrix.nonzero()
        edges = set(zip(rows, cols))
        self.assertIn((idx_b, idx_a), edges)
        self.assertEqual(matrix.nnz, 1)

    @patch("paperank.citation_matrix.get_citing_dois")
    @patch("paperank.citation_matrix.get_cited_dois")
    def test_include_citing_ignores_citers_not_in_list(self, mock_get_cited, mock_get_citing):
        # Arrange: citing list contains one DOI in-list (B) and one out-of-list (C)
        doi_a = "10.1000/a"
        doi_b = "10.1000/b"
        doi_c = "10.1000/c"  # not included in doi_list

        def citing_side_effect(doi):
            if doi == doi_a:
                return {"citing_dois": [doi_b, doi_c]}
            if doi == doi_b:
                return {"citing_dois": []}
            return {"citing_dois": []}

        mock_get_citing.side_effect = citing_side_effect
        mock_get_cited.return_value = {"cited_dois": []}

        # Act
        matrix, mapping = build_citation_sparse_matrix(
            [doi_a, doi_b], include_citing=True, max_workers=None, progress=False
        )

        # Assert: Only edge B -> A should be present; C is ignored (not in mapping)
        idx_a = mapping[doi_a]
        idx_b = mapping[doi_b]
        rows, cols = matrix.nonzero()
        edges = set(zip(rows, cols))
        self.assertEqual(edges, {(idx_b, idx_a)})
        self.assertEqual(matrix.nnz, 1)
        
if __name__ == "__main__":
    unittest.main()
