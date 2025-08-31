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

    def test_extract_authors_name_literal_given_family_precedence(self):
        meta = {
            "message": {
                "title": ["Title"],
                "issued": {"date-parts": [[2020]]},
                "author": [
                    # name should take precedence over literal and given/family
                    {
                        "name": "Dr. Example",
                        "literal": "ShouldNotBeUsed",
                        "given": "First",
                        "family": "Last",
                    },
                    # literal should be used when name is missing
                    {"literal": "Literal Used", "given": "X", "family": "Y"},
                    # fallback to given/family when name and literal missing
                    {"given": "OnlyGiven"},
                    {"family": "OnlyFamily"},
                    # should be skipped because all empty
                    {"given": "", "family": "   "},
                ],
            }
        }
        authors, title, year = extract_authors_title_year(meta)
        self.assertEqual(
            authors,
            ["Dr. Example", "Literal Used", "OnlyGiven", "OnlyFamily"],
        )
        self.assertEqual(title, "Title")
        self.assertEqual(year, 2020)

    def test_extract_authors_string_entries_and_skips_empty(self):
        meta = {
            "message": {
                "title": "Another",
                "issued": {"date-parts": [[1999, 1, 1]]},
                "author": [
                    "Grace Hopper",  # kept
                    "   ",  # skipped
                    "",  # skipped
                    {"given": "", "family": ""},  # skipped after strip
                    {"literal": "  "},  # skipped after strip
                    "String Author",  # kept
                ],
            }
        }
        authors, title, year = extract_authors_title_year(meta)
        self.assertEqual(authors, ["Grace Hopper", "String Author"])
        self.assertEqual(title, "Another")
        self.assertEqual(year, 1999)

    def test_extract_title_fallback_to_short_title(self):
        meta = {
            "message": {
                # No 'title' present; should fall back to short-title[0]
                "short-title": ["Shorty"],
                # Use alternate key 'date_parts' to ensure year extraction still works
                "issued": {"date_parts": [[2018]]},
                "author": [],
            }
        }
        authors, title, year = extract_authors_title_year(meta)
        self.assertEqual(authors, [])
        self.assertEqual(title, "Shorty")
        self.assertEqual(year, 2018)

    def test_extract_title_fallback_to_subtitle_when_title_and_short_title_missing(self):
        meta = {
            "message": {
                # No 'title' or 'short-title'; should fall back to subtitle[0]
                "subtitle": ["Subbed"],
                "published": {"date-parts": [[2005]]},
                "author": [],
            }
        }
        authors, title, year = extract_authors_title_year(meta)
        self.assertEqual(authors, [])
        self.assertEqual(title, "Subbed")
        self.assertEqual(year, 2005)

if __name__ == "__main__":
    unittest.main()
