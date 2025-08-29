import unittest
from unittest.mock import patch
import requests

from paperank.open_citations import get_citing_dois
from paperank.crossref import get_work_metadata, get_cited_dois

class TestNetworkErrors(unittest.TestCase):
    @patch("paperank.open_citations._get_session")
    def test_open_citations_http_error(self, mock_get_session):
        sess = mock_get_session.return_value
        resp = sess.get.return_value
        resp.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        with self.assertRaises(requests.HTTPError):
            get_citing_dois("10.1234/abc")

    @patch("paperank.crossref._get_session")
    def test_crossref_http_error(self, mock_get_session):
        sess = mock_get_session.return_value
        resp = sess.get.return_value
        resp.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        with self.assertRaises(requests.HTTPError):
            get_work_metadata("10.1234/def")

    @patch("paperank.crossref.get_work_metadata")
    def test_get_cited_dois_handles_missing_reference(self, mock_meta):
        mock_meta.return_value = {"message": {"reference": [{}, {"DOI": "10.1000/xyz"}]}}
        res = get_cited_dois("10.1234/ghi")
        self.assertEqual(res["article_doi"], "10.1234/ghi")
        self.assertEqual(res["cited_dois"], ["10.1000/xyz"]) 

if __name__ == "__main__":
    unittest.main()
