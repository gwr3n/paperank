import unittest

from paperank.crossref import extract_authors_title_year


class TestCrossrefExtract(unittest.TestCase):
    def test_extract_authors_title_year_partial_envelope(self):
        meta = {
            "message": {
                "title": ["An example"],
                "issued": {"date-parts": [[2021, 5, 1]]},
                "author": [
                    {"given": "Ada", "family": "Lovelace"},
                    {"given": "Alan", "family": "Turing"},
                ],
            }
        }
        authors, title, year = extract_authors_title_year(meta)
        self.assertEqual(authors, ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(title, "An example")
        self.assertEqual(year, 2021)


if __name__ == "__main__":
    unittest.main()
