import unittest
from unittest.mock import patch

from paperank.citation_crawler import (
    _with_progress,  # added for testing tqdm path
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

    def test__with_progress_uses_tqdm_when_available(self):
        import types

        # Create a fake tqdm module with a callable 'tqdm' that records args
        class DummyTqdm:
            def __init__(self):
                self.calls = []

            def __call__(self, iterable, desc=None, leave=None):
                items = list(iterable)
                self.calls.append({"desc": desc, "leave": leave, "items": items})
                for x in items:
                    yield x

        dummy = DummyTqdm()
        fake_module = types.ModuleType("tqdm")
        fake_module.tqdm = dummy

        with patch.dict("sys.modules", {"tqdm": fake_module}):
            data = [1, 2, 3]
            wrapped = _with_progress(data, enabled=True, desc="My progress")
            # Ensure it iterates the same items and used our dummy tqdm
            self.assertEqual(list(wrapped), data)
            self.assertTrue(dummy.calls, "tqdm should have been called")
            self.assertEqual(dummy.calls[0]["desc"], "My progress")
            self.assertEqual(dummy.calls[0]["leave"], False)
            self.assertEqual(dummy.calls[0]["items"], data)

    def test__with_progress_graceful_fallback_when_tqdm_missing(self):
        # Force ImportError for tqdm to take the except path and return the original iterable
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tqdm":
                raise ImportError("forced-missing-tqdm")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            data = [4, 5]
            wrapped = _with_progress(data, enabled=True, desc="Ignored")
            # Should return the original iterable unchanged
            self.assertIs(wrapped, data)
            self.assertEqual(list(wrapped), data)


if __name__ == "__main__":
    unittest.main()
