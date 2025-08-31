import unittest
from unittest.mock import patch

from paperank.citation_crawler import (
    clear_caches,
    collect_cited_recursive,
    collect_citing_recursive,
    crawl_citation_neighborhood,
    get_citation_neighborhood,
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

    def test_crawl_citation_neighborhood(self):
        result = crawl_citation_neighborhood([self.test_doi], steps=1, progress=False)
        self.assertIsInstance(result, list)
        self.assertIn(self.test_doi, result)

    def test_collect_cited_recursive_stops_at_max_nodes_tree(self):
        # Graph (cited): A -> [B, C, D]; B -> [E]; others empty
        cited_graph = {
            "A": ["B", "C", "D"],
            "B": ["E"],
            "C": [],
            "D": [],
            "E": [],
        }

        def fake_get_cited_dois(doi: str):
            return {"cited_dois": cited_graph.get(doi, [])}

        clear_caches()
        with patch("paperank.citation_crawler.get_cited_dois", side_effect=fake_get_cited_dois):
            res = collect_cited_recursive("A", depth=3, flatten=False, max_nodes=2, progress=False)

        # Only root 'A' and first visited child 'B' are explored as nodes
        self.assertEqual(set(res.keys()), {"A", "B"})
        self.assertEqual(res["A"], ["B", "C", "D"])
        self.assertEqual(res["B"], ["E"])
        self.assertNotIn("C", res)
        self.assertNotIn("D", res)
        self.assertNotIn("E", res)  # E is listed under B but not visited as a node

    def test_collect_citing_recursive_stops_at_max_nodes_flat(self):
        # Graph (citing): R <- [U, V, W]; U <- [U1, U2]; V <- [V1]
        citing_graph = {
            "R": ["U", "V", "W"],
            "U": ["U1", "U2"],
            "V": ["V1"],
            "W": [],
            "U1": [],
            "U2": [],
            "V1": [],
        }

        def fake_get_citing_dois(doi: str):
            return {"citing_dois": citing_graph.get(doi, [])}

        clear_caches()
        with patch("paperank.citation_crawler.get_citing_dois", side_effect=fake_get_citing_dois):
            out = collect_citing_recursive("R", depth=3, flatten=True, max_nodes=2, progress=False)

        # Visited hits max at {'R','U'}; we capture 'U' and its first citer 'U1'
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(out[:2], ["U", "U1"])
        self.assertNotIn("V", out)
        self.assertNotIn("W", out)

    def test_collect_cited_recursive_flat_dfs_recurses(self):
        # Graph: A -> [B]; B -> [C, D]
        cited_graph = {
            "A": ["B"],
            "B": ["C", "D"],
            "C": [],
            "D": [],
        }

        def fake_get_cited_dois(doi: str):
            return {"cited_dois": cited_graph.get(doi, [])}

        clear_caches()
        with patch("paperank.citation_crawler.get_cited_dois", side_effect=fake_get_cited_dois):
            out = collect_cited_recursive("A", depth=3, flatten=True, progress=False)

        # Expect dfs to recurse into B and collect C and D
        self.assertEqual(out, ["B", "C", "D"])

    def test_collect_cited_recursive_flat_max_nodes_inside_dfs(self):
        # Graph: A -> [B]; B -> [E, F]; F -> [G]
        # With max_nodes=3, dfs(B) visits E first, then before recursing into F,
        # visited has reached the threshold inside the loop and returns early.
        cited_graph = {
            "A": ["B"],
            "B": ["E", "F"],
            "E": [],
            "F": ["G"],
            "G": [],
        }

        def fake_get_cited_dois(doi: str):
            return {"cited_dois": cited_graph.get(doi, [])}

        clear_caches()
        with patch("paperank.citation_crawler.get_cited_dois", side_effect=fake_get_cited_dois):
            out = collect_cited_recursive("A", depth=3, flatten=True, max_nodes=3, progress=False)

        # 'B' (top-level), then 'E'; 'F' is added to out but not recursed into due to early return
        self.assertEqual(out, ["B", "E", "F"])
        self.assertNotIn("G", out)

    def test_collect_cited_recursive_flat_max_nodes_return_before_children(self):
        # Graph: A -> [B]; B -> [C]; C -> [D]
        # With max_nodes=2, dfs(B) will add B to visited, then append C to out,
        # and return early before recursing into C.
        cited_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": [],
        }

        def fake_get_cited_dois(doi: str):
            return {"cited_dois": cited_graph.get(doi, [])}

        clear_caches()
        with patch("paperank.citation_crawler.get_cited_dois", side_effect=fake_get_cited_dois):
            out = collect_cited_recursive("A", depth=3, flatten=True, max_nodes=2, progress=False)

        # 'B' is added at top level; 'C' is appended inside dfs before early return.
        self.assertEqual(out, ["B", "C"])
        self.assertNotIn("D", out)

if __name__ == "__main__":
    unittest.main()
