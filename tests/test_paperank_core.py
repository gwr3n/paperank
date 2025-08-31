import io
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from paperank.paperank_core import (
    crawl_and_rank_bidirectional_neighborhood,
    crawl_and_rank_frontier,
    rank,
    rank_and_save_publications_CSV,
    rank_and_save_publications_JSON,
)


class TestPapeRankCore(unittest.TestCase):
    def setUp(self):
        # Use a small set of DOIs for testing
        self.doi_list = ["10.1016/j.ejor.2016.12.001", "10.1080/1540496x.2019.1696189", "10.1016/j.intfin.2017.09.008"]
        self.alpha = 0.85

    def test_rank_returns_sorted_list(self):
        ranked = rank(self.doi_list, alpha=self.alpha, progress=False)
        self.assertIsInstance(ranked, list)
        self.assertTrue(all(isinstance(x, tuple) and len(x) == 2 for x in ranked))
        # Scores should be sorted descending
        scores = [score for _, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_and_save_publications_JSON(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            rank_and_save_publications_JSON(self.doi_list, out_path, alpha=self.alpha, max_results=2, progress=False)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                data = f.read()
            self.assertIn('"items"', data)
            self.assertIn('"doi"', data)

    def test_rank_and_save_publications_CSV(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.csv")
            rank_and_save_publications_CSV(self.doi_list, out_path, alpha=self.alpha, max_results=2, progress=False)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertTrue(lines[0].startswith("rank,doi,score,authors,title,year"))
            self.assertTrue(any("10.1016/j.ejor.2016.12.001" in line for line in lines))

    @patch("paperank.paperank_core.get_work_metadata")
    @patch("paperank.paperank_core.extract_authors_title_year")
    def test_rank_and_save_publications_CSV_quoting(self, mock_extract, mock_meta):
        # Provide titles/authors with commas, quotes, and newlines
        mock_meta.return_value = {"message": {}}
        mock_extract.side_effect = [
            (["Doe, John"], 'A "Complex" Title, With Commas', 2020),
            (["Alice\nBob"], "Another title", None),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.csv")
            rank_and_save_publications_CSV(self.doi_list, out_path, alpha=self.alpha, max_results=2, progress=False)
            with open(out_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Expect quotes to be doubled and fields quoted
            self.assertIn('"A ""Complex"" Title, With Commas"', text)
            self.assertIn('"Doe, John"', text)
            self.assertIn('"Alice\nBob"', text)


class TestPapeRankCoreCrawlers(unittest.TestCase):
    def test_rank_pipeline_and_debug_tqdm(self):
        # Arrange
        doi_list = ["a", "b", "c"]

        class FakeSparse:
            def __init__(self, shape, nnz):
                self.shape = shape
                self.nnz = nnz

        fake_adj = FakeSparse((3, 3), 2)
        fake_S = FakeSparse((3, 3), 3)

        with (
            patch("paperank.paperank_core.build_citation_sparse_matrix") as p_build,
            patch("paperank.paperank_core.adjacency_to_stochastic_matrix") as p_adj2stoch,
            patch("paperank.paperank_core.compute_publication_rank_teleport") as p_pr,
        ):

            p_build.return_value = (fake_adj, {"a": 0, "b": 1, "c": 2})
            p_adj2stoch.return_value = fake_S
            # scores -> expect order b (0.5), c (0.3), a (0.2)
            p_pr.return_value = np.array([0.2, 0.5, 0.3], dtype=float)

            # Act
            res = rank(
                doi_list,
                alpha=0.9,
                debug=True,  # should force progress="tqdm"
                progress=False,
                tol=1e-6,
                max_iter=50,
                teleport=np.array([1 / 3, 1 / 3, 1 / 3]),
            )

            # Assert ranking and pipeline calls
            self.assertEqual(res, [("b", 0.5), ("c", 0.3), ("a", 0.2)])
            p_build.assert_called_once()
            args_build, kwargs_build = p_build.call_args
            self.assertEqual(list(args_build[0]), doi_list)
            self.assertEqual(kwargs_build.get("progress", None), False)

            p_adj2stoch.assert_called_once_with(fake_adj)

            # compute_publication_rank_teleport received "tqdm" progress
            args_pr, kwargs_pr = p_pr.call_args
            self.assertIs(args_pr[0], fake_S)
            self.assertEqual(kwargs_pr["alpha"], 0.9)
            self.assertEqual(kwargs_pr["tol"], 1e-6)
            self.assertEqual(kwargs_pr["max_iter"], 50)
            self.assertEqual(kwargs_pr["progress"], "tqdm")

    def test_crawl_and_rank_bidirectional_neighborhood_json_and_warning(self):
        doi = "10.1000/xyz.123"
        doi_filename = "10_1000_xyz_123"

        with (
            patch("paperank.paperank_core.get_citation_neighborhood") as p_neigh,
            patch("paperank.paperank_core.rank_and_save_publications_JSON") as p_save_json,
            patch("paperank.paperank_core.rank") as p_rank,
        ):

            p_neigh.return_value = ["a", "b"]
            p_rank.return_value = [("a", 0.7), ("b", 0.3)]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = crawl_and_rank_bidirectional_neighborhood(
                    doi,
                    forward_steps=2,
                    backward_steps=3,
                    alpha=0.75,
                    output_format="json",
                    debug=True,
                    progress=True,
                    tol=1e-9,
                    max_iter=123,
                    teleport=None,
                )
                self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

            p_neigh.assert_called_once_with(doi, forward_steps=2, backward_steps=3)
            p_save_json.assert_called_once()
            args, kwargs = p_save_json.call_args
            self.assertEqual(list(args[0]), ["a", "b"])
            self.assertEqual(kwargs["out_path"], f"{doi_filename}.json")
            self.assertEqual(kwargs["alpha"], 0.75)
            self.assertEqual(kwargs["tol"], 1e-9)
            self.assertEqual(kwargs["max_iter"], 123)
            self.assertTrue(kwargs["teleport"] is None)
            self.assertTrue(kwargs["progress"])

            p_rank.assert_called_once()
            _, kwargs_rank = p_rank.call_args
            self.assertEqual(kwargs_rank["alpha"], 0.75)
            self.assertTrue(kwargs_rank["debug"])
            # In this deprecated function, progress is passed through unchanged
            self.assertTrue(kwargs_rank["progress"])
            self.assertEqual(result, [("a", 0.7), ("b", 0.3)])

    def test_crawl_and_rank_bidirectional_neighborhood_csv_and_unknown_format(self):
        doi = "10.1/abc.def"
        doi_filename = "10_1_abc_def"

        with (
            patch("paperank.paperank_core.get_citation_neighborhood") as p_neigh,
            patch("paperank.paperank_core.rank_and_save_publications_CSV") as p_save_csv,
            patch("paperank.paperank_core.rank") as p_rank,
        ):
            p_neigh.return_value = ["x", "y"]
            p_rank.return_value = [("x", 1.0), ("y", 0.5)]

            # CSV branch
            res_csv = crawl_and_rank_bidirectional_neighborhood(doi, output_format="csv")
            self.assertEqual(res_csv, [("x", 1.0), ("y", 0.5)])
            args, kwargs = p_save_csv.call_args
            self.assertEqual(list(args[0]), ["x", "y"])
            self.assertEqual(kwargs["out_path"], f"{doi_filename}.csv")

            # Unknown format prints a message and still returns rank
            buf = io.StringIO()
            with patch("sys.stdout", new=buf):
                res_unknown = crawl_and_rank_bidirectional_neighborhood(doi, output_format="xml")
            self.assertIn("Unknown output format", buf.getvalue())
            self.assertEqual(res_unknown, [("x", 1.0), ("y", 0.5)])

    def test_crawl_and_rank_frontier_with_str_seed_json(self):
        doi = "10.1000/xyz.123"
        base = "10_1000_xyz_123"

        with (
            patch("paperank.paperank_core.crawl_citation_neighborhood") as p_crawl,
            patch("paperank.paperank_core.rank_and_save_publications_JSON") as p_save_json,
            patch("paperank.paperank_core.rank") as p_rank,
        ):

            p_crawl.return_value = ["p", "q", "r"]
            p_rank.return_value = [("p", 0.4), ("q", 0.35), ("r", 0.25)]

            result = crawl_and_rank_frontier(
                doi,
                steps=2,
                min_year=2010,
                min_citations=50,
                alpha=0.8,
                output_format="json",
                debug=False,
                progress=False,
                tol=1e-8,
                max_iter=250,
                teleport=None,
            )

            p_crawl.assert_called_once()
            args, kwargs = p_crawl.call_args
            self.assertEqual(list(args[0]), [doi])  # seeds normalized to list
            self.assertEqual(kwargs["steps"], 2)
            self.assertEqual(kwargs["min_year"], 2010)
            self.assertEqual(kwargs["min_citations"], 50)
            self.assertFalse(kwargs["progress"])

            p_save_json.assert_called_once()
            _, kwargs = p_save_json.call_args
            self.assertEqual(kwargs["out_path"], f"{base}.json")
            self.assertEqual(kwargs["alpha"], 0.8)
            self.assertEqual(kwargs["tol"], 1e-8)
            self.assertEqual(kwargs["max_iter"], 250)
            self.assertFalse(kwargs["progress"])

            # rank called with progress unchanged when debug=False
            _, kwargs_rank = p_rank.call_args
            self.assertFalse(kwargs_rank["progress"])
            self.assertEqual(result, [("p", 0.4), ("q", 0.35), ("r", 0.25)])

    def test_crawl_and_rank_frontier_with_list_seeds_csv_and_debug_tqdm(self):
        seeds = ["a", "b", "a"]  # contains duplicate; should deduplicate to ["a", "b"]

        with (
            patch("paperank.paperank_core.crawl_citation_neighborhood") as p_crawl,
            patch("paperank.paperank_core.rank_and_save_publications_CSV") as p_save_csv,
            patch("paperank.paperank_core.rank") as p_rank,
        ):

            p_crawl.return_value = ["u", "v"]
            p_rank.return_value = [("v", 0.6), ("u", 0.4)]

            result = crawl_and_rank_frontier(
                seeds,
                steps=1,
                alpha=0.85,
                output_format="csv",
                debug=True,  # should switch rank progress to "tqdm"
                progress=True,
            )

            # Crawl called with deduplicated seeds in order
            args, kwargs = p_crawl.call_args
            self.assertEqual(list(args[0]), ["a", "b"])
            self.assertEqual(kwargs["steps"], 1)
            self.assertTrue(kwargs["progress"])

            # Save CSV path uses "crawl_{len(seeds)}_seeds.csv"
            args_save, kwargs_save = p_save_csv.call_args
            self.assertEqual(kwargs_save["out_path"], "crawl_2_seeds.csv")
            # doi_list is the first positional argument
            self.assertEqual(list(args_save[0]), ["u", "v"])
            # Save uses original progress
            self.assertTrue(kwargs_save["progress"])

            # rank uses "tqdm" due to debug=True
            _, kwargs_rank = p_rank.call_args
            self.assertEqual(kwargs_rank["progress"], "tqdm")
            self.assertEqual(result, [("v", 0.6), ("u", 0.4)])


if __name__ == "__main__":
    unittest.main()
