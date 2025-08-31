import os
import tempfile
import unittest
from unittest.mock import patch

from paperank.paperank_core import rank, rank_and_save_publications_CSV, rank_and_save_publications_JSON


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


if __name__ == "__main__":
    unittest.main()
