import io
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import scipy.sparse

from paperank.paperank_matrix import (
    adjacency_to_stochastic_matrix,
    apply_random_jump,
    compute_publication_rank,
    compute_publication_rank_teleport,
)


class TestPapeRankMatrix(unittest.TestCase):
    def get_sample_adjacency_matrix(self):
        doi_to_index = {
            "10.1016/j.ejor.2016.12.001": 0,
            "10.1080/1540496x.2019.1696189": 1,
            "10.1016/j.intfin.2017.09.008": 2,
            "10.3390/s24155046": 3,
            "10.3917/reco.724.0591": 4,
            "10.1080/1540496x.2018.1540349": 5,
            "10.3390/diagnostics15030313": 6,
            "10.1108/ijoem-08-2021-1232": 7,
            "10.3390/ijfs10040110": 8,
            "10.3390/photonics11100964": 9,
            "10.1016/j.pacfin.2020.101328": 10,
            "10.3390/su12166325": 11,
            "10.2139/ssrn.4884905": 12,
            "10.1063/5.0226761": 13,
            "10.1016/j.ejor.2025.01.001": 14,
            "10.3390/su15054508": 15,
            "10.1007/s11156-020-00884-y": 16,
            "10.1016/j.optcom.2020.126513": 17,
            "10.1155/2020/4294538": 18,
            "10.1016/j.jbusres.2019.09.046": 19,
            "10.1016/j.cageo.2019.01.016": 20,
            "10.1596/978-1-4648-1447-1_bm": 21,
            "10.21511/imfi.20(2).2023.26": 22,
            "10.1016/j.ejor.2025.01.036": 23,
            "10.1016/j.jfs.2025.101394": 24,
            "10.1080/23322039.2023.2256127": 25,
            "10.1002/csr.2567": 26,
            "10.1177/18479790231165603": 27,
            "10.3917/reco.pr2.0162": 28,
            "10.1016/j.eswa.2024.125004": 29,
            "10.3390/math12050780": 30,
            "10.1007/s11156-018-0712-y": 31,
            "10.1007/978-981-16-3509-0_3": 32,
            "10.1016/j.irfa.2022.102044": 33,
            "10.1111/saje.12395": 34,
            "10.1109/cisp-bmei60920.2023.10373246": 35,
            "10.1108/jfep-06-2021-0162": 36,
            "10.38016/jista.1566965": 37,
            "10.2139/ssrn.4683333": 38,
            "10.1002/ldr.5015": 39,
            "10.1093/ijlct/ctad135": 40,
            "10.3390/s23042096": 41,
            "10.1016/j.ecolind.2018.10.025": 42,
            "10.1016/j.qref.2024.101919": 43,
            "10.1016/j.optlastec.2024.111730": 44,
            "10.1108/cfri-08-2021-0166": 45,
            "10.1016/j.intfin.2018.04.005": 46,
            "10.1016/j.fuel.2019.116167": 47,
            "10.1109/sami63904.2025.10883055": 48,
            "10.1007/978-981-16-3509-0_2": 49,
            "10.3390/app14041455": 50,
            "10.1145/3278252.3278266": 51,
            "10.37394/232029.2023.2.8": 52,
            "10.2139/ssrn.3806368": 53,
            "10.22495/rgcv10i1p6": 54,
            "10.2139/ssrn.3550458": 55,
            "10.1093/wbro/lkaa006": 56,
            "10.1016/j.ribaf.2019.101177": 57,
            "10.1080/1540496x.2017.1411257": 58,
            "10.1111/ajfs.12367": 59,
            "10.1016/j.ememar.2022.100963": 60,
            "10.1093/rof/rfu049": 61,
            "10.21034/qr.1021": 62,
            "10.1016/0304-3932(93)90016-9": 63,
            "10.1093/rof/rft039": 64,
            "10.1016/j.omega.2009.09.007": 65,
            "10.1080/03610929108830707": 66,
            "10.1214/aos/1176344552": 67,
            "10.1016/j.jimonfin.2014.02.008": 68,
            "10.1016/j.jbankfin.2010.10.005": 69,
            "10.2307/1992111": 70,
            "10.1016/j.jfineco.2010.02.008": 71,
            "10.1016/j.omega.2009.10.009": 72,
            "10.2307/2334192": 73,
            "10.1016/j.ejor.2006.08.043": 74,
            "10.1016/j.jfineco.2008.09.003": 75,
            "10.1016/j.intfin.2013.01.004": 76,
            "10.1016/j.frl.2015.01.001": 77,
            "10.1016/j.jbankfin.2007.01.004": 78,
            "10.1016/j.ejor.2016.06.035": 79,
            "10.1080/00207543.2014.917771": 80,
            "10.1016/j.ejor.2014.06.007": 81,
            "10.2307/1907413": 82,
            "10.1080/13504851.2011.558474": 83,
            "10.1016/j.ejor.2015.09.057": 84,
            "10.1016/j.jbankfin.2006.11.003": 85,
        }
        citation_edges = [
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
            (0, 80),
            (0, 81),
            (0, 82),
            (0, 83),
            (0, 84),
            (0, 85),
            (1, 0),
            (1, 75),
            (2, 0),
            (2, 68),
            (2, 78),
            (3, 0),
            (4, 0),
            (4, 19),
            (4, 68),
            (4, 75),
            (4, 77),
            (4, 82),
            (4, 83),
            (5, 0),
            (5, 2),
            (5, 63),
            (5, 75),
            (5, 85),
            (6, 0),
            (7, 0),
            (7, 36),
            (7, 45),
            (7, 56),
            (8, 0),
            (8, 10),
            (8, 63),
            (8, 69),
            (8, 70),
            (8, 76),
            (8, 85),
            (9, 0),
            (9, 17),
            (10, 0),
            (10, 63),
            (10, 69),
            (10, 75),
            (11, 0),
            (11, 74),
            (12, 0),
            (12, 70),
            (12, 77),
            (13, 0),
            (14, 0),
            (15, 0),
            (15, 71),
            (15, 75),
            (15, 76),
            (16, 0),
            (16, 68),
            (17, 0),
            (18, 0),
            (19, 0),
            (19, 68),
            (19, 75),
            (19, 77),
            (19, 82),
            (19, 83),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
            (23, 70),
            (23, 76),
            (24, 0),
            (24, 75),
            (25, 0),
            (25, 71),
            (25, 75),
            (25, 77),
            (26, 0),
            (26, 61),
            (26, 68),
            (26, 75),
            (26, 78),
            (27, 0),
            (27, 11),
            (27, 74),
            (28, 0),
            (28, 19),
            (28, 68),
            (28, 75),
            (28, 77),
            (28, 82),
            (28, 83),
            (29, 0),
            (30, 0),
            (31, 0),
            (31, 69),
            (31, 77),
            (32, 0),
            (32, 2),
            (32, 63),
            (32, 75),
            (32, 85),
            (33, 0),
            (33, 75),
            (33, 82),
            (34, 0),
            (34, 76),
            (34, 77),
            (35, 0),
            (35, 77),
            (36, 0),
            (36, 45),
            (37, 0),
            (38, 0),
            (38, 68),
            (38, 75),
            (38, 76),
            (39, 0),
            (40, 0),
            (41, 0),
            (42, 0),
            (43, 0),
            (43, 70),
            (43, 76),
            (43, 77),
            (43, 83),
            (44, 0),
            (44, 17),
            (45, 0),
            (46, 0),
            (46, 69),
            (46, 76),
            (47, 0),
            (47, 81),
            (48, 0),
            (49, 0),
            (49, 75),
            (50, 0),
            (51, 0),
            (51, 77),
            (52, 0),
            (53, 0),
            (53, 46),
            (53, 75),
            (54, 0),
            (56, 0),
            (57, 0),
            (58, 0),
            (58, 61),
            (58, 68),
            (58, 71),
            (58, 75),
            (58, 78),
            (59, 0),
            (60, 0),
            (60, 69),
            (60, 77),
            (64, 75),
            (65, 74),
            (68, 69),
            (68, 75),
            (68, 76),
            (68, 78),
            (68, 85),
            (71, 75),
            (71, 82),
            (72, 74),
            (75, 82),
            (76, 70),
            (76, 82),
            (76, 83),
            (76, 85),
            (77, 70),
            (77, 71),
            (77, 75),
            (77, 76),
            (77, 82),
            (83, 70),
            (83, 82),
            (84, 70),
            (84, 71),
            (84, 75),
            (84, 77),
            (85, 82),
        ]
        num_nodes = len(doi_to_index)
        adjacency_matrix = scipy.sparse.lil_matrix((num_nodes, num_nodes))
        for src, tgt in citation_edges:
            adjacency_matrix[src, tgt] = 1
        return adjacency_matrix, doi_to_index

    def test_adjacency_to_stochastic_matrix(self):
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        stochastic_matrix = adjacency_to_stochastic_matrix(adjacency_matrix)
        row_sums = np.array(stochastic_matrix.sum(axis=1)).flatten()
        for i in range(stochastic_matrix.shape[0]):
            if adjacency_matrix[i].sum() > 0:
                self.assertTrue(np.isclose(row_sums[i], 1.0), f"Row {i} does not sum to 1")
            else:
                self.assertEqual(row_sums[i], 0, f"Row {i} should sum to 0")

    def test_stochastic_matrix_with_random_jump(self):
        alpha = 0.85
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        stochastic_matrix = adjacency_to_stochastic_matrix(adjacency_matrix)
        jump_matrix = apply_random_jump(stochastic_matrix, alpha=alpha)
        row_sums = np.array(jump_matrix.sum(axis=1)).flatten()
        for i in range(jump_matrix.shape[0]):
            self.assertTrue(np.isclose(row_sums[i], 1.0), f"Row {i} does not sum to 1 after random jump")

    def test_paperank(self):
        alpha = 0.85
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        S = adjacency_to_stochastic_matrix(adjacency_matrix)
        G = apply_random_jump(S, alpha=alpha)
        r = compute_publication_rank(G, tol=1e-12, max_iter=10000)
        self.assertTrue(r.ndim == 1 and r.shape[0] == G.shape[0], "Rank vector has wrong shape")
        self.assertTrue(np.all(r >= -1e-15), "Rank vector has negative entries")
        self.assertTrue(np.isclose(r.sum(), 1.0, atol=1e-10), "Rank vector does not sum to 1")
        r_next = G.transpose().tocsr() @ r
        r_next_sum = r_next.sum()
        if r_next_sum != 0:
            r_next /= r_next_sum
        self.assertTrue(np.allclose(r_next, r, atol=1e-8), "r is not stationary for G")

    def test_empty_adjacency_matrix(self):
        # Test with an empty adjacency matrix
        size = 5
        adjacency_matrix = scipy.sparse.lil_matrix((size, size))
        stochastic_matrix = adjacency_to_stochastic_matrix(adjacency_matrix)
        self.assertTrue(np.all(stochastic_matrix.sum(axis=1) == 0))
        jump_matrix = apply_random_jump(stochastic_matrix, alpha=0.85)
        row_sums = np.array(jump_matrix.sum(axis=1)).flatten()
        self.assertTrue(np.allclose(row_sums, 1.0), "All rows should sum to 1 after random jump even if empty")

    def test_single_node_matrix(self):
        # Test with a single node (self-loop)
        adjacency_matrix = scipy.sparse.lil_matrix((1, 1))
        adjacency_matrix[0, 0] = 1
        stochastic_matrix = adjacency_to_stochastic_matrix(adjacency_matrix)
        self.assertTrue(np.isclose(stochastic_matrix[0, 0], 1.0))
        jump_matrix = apply_random_jump(stochastic_matrix, alpha=0.85)
        self.assertTrue(np.isclose(jump_matrix[0, 0], 1.0))
        r = compute_publication_rank(jump_matrix)
        self.assertTrue(np.isclose(r[0], 1.0))

    def test_non_square_matrix(self):
        # Should raise an error if matrix is not square
        adjacency_matrix = scipy.sparse.lil_matrix((3, 4))
        with self.assertRaises(Exception):
            adjacency_to_stochastic_matrix(adjacency_matrix)

    def test_apply_random_jump_alpha_bounds(self):
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        stochastic_matrix = adjacency_to_stochastic_matrix(adjacency_matrix)
        # Alpha = 0 (pure random jump)
        jump_matrix = apply_random_jump(stochastic_matrix, alpha=0.0)
        row_sums = np.array(jump_matrix.sum(axis=1)).flatten()
        self.assertTrue(np.allclose(row_sums, 1.0))
        # Alpha = 1 (no random jump)
        jump_matrix = apply_random_jump(stochastic_matrix, alpha=1.0)
        row_sums = np.array(jump_matrix.sum(axis=1)).flatten()
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_compute_publication_rank_with_progress(self):
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        S = adjacency_to_stochastic_matrix(adjacency_matrix)
        G = apply_random_jump(S, alpha=0.85)
        # Use integer progress (prints every N iterations)
        r = compute_publication_rank(G, tol=1e-12, max_iter=100, progress=10)
        self.assertTrue(np.isclose(r.sum(), 1.0, atol=1e-10))
        # Use tqdm progress bar (if tqdm is installed)
        try:
            r2 = compute_publication_rank(G, tol=1e-12, max_iter=100, progress="tqdm")
            self.assertTrue(np.isclose(r2.sum(), 1.0, atol=1e-10))
        except Exception:
            pass  # tqdm not installed, skip

    def test_compute_publication_rank_with_callback(self):
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        S = adjacency_to_stochastic_matrix(adjacency_matrix)
        G = apply_random_jump(S, alpha=0.85)
        # Stop after 5 iterations using callback
        call_counter = {"count": 0}

        def cb(iteration, delta, r):
            call_counter["count"] += 1
            return iteration >= 5

        r = compute_publication_rank(G, tol=1e-12, max_iter=100, callback=cb)
        self.assertTrue(call_counter["count"] >= 5)
        self.assertTrue(np.isclose(r.sum(), 1.0, atol=1e-10))

    def test_compute_publication_rank_with_nonuniform_init(self):
        adjacency_matrix, _ = self.get_sample_adjacency_matrix()
        S = adjacency_to_stochastic_matrix(adjacency_matrix)
        G = apply_random_jump(S, alpha=0.85)
        # Initial vector: all probability on node 0
        init = np.zeros(G.shape[0])
        init[0] = 1.0
        r = compute_publication_rank(G, tol=1e-12, max_iter=100, init=init)
        self.assertTrue(np.isclose(r.sum(), 1.0, atol=1e-10))
        self.assertTrue(r[0] < 1.0)  # Should have spread out

    def test_numpy_array_input(self):
        # Integer numpy adjacency with a dangling row
        A = np.array(
            [
                [0, 1, 1],
                [0, 0, 0],  # dangling
                [1, 0, 0],
            ],
            dtype=int,
        )
        S = adjacency_to_stochastic_matrix(A)
        self.assertIsInstance(S, scipy.sparse.csr_matrix)
        row_sums = np.array(S.sum(axis=1)).ravel()
        self.assertTrue(np.isclose(row_sums[0], 1.0))
        self.assertTrue(np.isclose(row_sums[1], 0.0))
        self.assertTrue(np.isclose(row_sums[2], 1.0))

    def test_teleport_on_all_dangling(self):
        # All-zero (all dangling) matrix should yield teleport vector as stationary distribution
        n = 5
        S = scipy.sparse.csr_matrix((n, n))
        v = np.array([0.10, 0.20, 0.30, 0.40, 0.0], dtype=float)
        v /= v.sum()
        r = compute_publication_rank_teleport(S, alpha=0.85, teleport=v, tol=1e-14, max_iter=100)
        self.assertTrue(np.allclose(r, v, atol=1e-12))

    def test_custom_teleport_influences_rank(self):
        # Small chain graph: 0->1, 1->2
        rows = np.array([0, 1])
        cols = np.array([1, 2])
        data = np.ones_like(rows)
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
        S = adjacency_to_stochastic_matrix(A)
        r_uniform = compute_publication_rank_teleport(S, alpha=0.85, teleport=None, tol=1e-14, max_iter=1000)
        v = np.array([0.8, 0.1, 0.1], dtype=float)
        v /= v.sum()
        r_biased = compute_publication_rank_teleport(S, alpha=0.85, teleport=v, tol=1e-14, max_iter=1000)
        self.assertTrue(np.isclose(r_uniform.sum(), 1.0))
        self.assertTrue(np.isclose(r_biased.sum(), 1.0))
        # Teleport bias should change the stationary distribution
        self.assertFalse(np.allclose(r_uniform, r_biased))

    def test_negative_entries_raise(self):
        # Build a small adjacency with a negative entry
        rows = np.array([0, 1])
        cols = np.array([1, 0])
        data = np.array([1.0, -1.0])  # negative entry
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(2, 2))
        with self.assertRaises(ValueError):
            adjacency_to_stochastic_matrix(A)

    def test_apply_random_jump_deprecation_warning(self):
        # Small matrix is fine; ensure deprecation warning is emitted
        A = np.array([[0, 1], [1, 0]], dtype=int)
        S = adjacency_to_stochastic_matrix(A)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = apply_random_jump(S, alpha=0.85)
            self.assertTrue(any(issubclass(wi.category, DeprecationWarning) for wi in w))

    def test_teleport_convergence_small_graph(self):
        # Triangle graph 0->1,1->2,2->0
        rows = np.array([0, 1, 2])
        cols = np.array([1, 2, 0])
        data = np.ones(3)
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))
        S = adjacency_to_stochastic_matrix(A)
        r = compute_publication_rank_teleport(S, alpha=0.85, tol=1e-14, max_iter=1000)
        self.assertTrue(np.isclose(r.sum(), 1.0))
        # By symmetry, ranks should be equal
        self.assertTrue(np.allclose(r, np.full(3, 1 / 3)))

    def test_apply_random_jump_dense_warning_estimate(self):
        # Moderate N to trigger memory warning message
        n = 100
        A = scipy.sparse.csr_matrix((n, n))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = apply_random_jump(adjacency_to_stochastic_matrix(A), alpha=0.85)
            # Either DeprecationWarning or user warning for dense materialization expected
            cats = [wi.category for wi in w]
            self.assertTrue(any(issubclass(c, (DeprecationWarning, UserWarning)) for c in cats))

    # ===== Additional tests for compute_publication_rank_teleport error paths =====

    def test_teleport_alpha_out_of_range_raises(self):
        S = scipy.sparse.identity(2, format="csr", dtype=float)
        with self.assertRaisesRegex(ValueError, r"alpha must be in \[0, 1\]"):
            compute_publication_rank_teleport(S, alpha=-0.1)
        with self.assertRaisesRegex(ValueError, r"alpha must be in \[0, 1\]"):
            compute_publication_rank_teleport(S, alpha=1.1)

    def test_teleport_non_square_matrix_raises(self):
        S = scipy.sparse.csr_matrix(np.ones((2, 3), dtype=float))
        with self.assertRaisesRegex(ValueError, r"must be square"):
            compute_publication_rank_teleport(S)

    def test_teleport_empty_matrix_returns_empty_array(self):
        S = scipy.sparse.csr_matrix((0, 0), dtype=float)
        r = compute_publication_rank_teleport(S)
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (0,))

    def test_teleport_negative_entries_raise_value_error_sparse_and_dense(self):
        # Sparse negative entry
        data = np.array([1.0, -0.5, 0.5], dtype=float)
        rows = np.array([0, 0, 1])
        cols = np.array([0, 1, 1])
        S_neg_sparse = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(2, 2))
        with self.assertRaisesRegex(ValueError, r"must be non-negative"):
            compute_publication_rank_teleport(S_neg_sparse)

        # Dense negative entry
        S_neg_dense = np.array([[1.0, -0.1], [0.0, 0.0]], dtype=float)
        with self.assertRaisesRegex(ValueError, r"must be non-negative"):
            compute_publication_rank_teleport(S_neg_dense)

    def test_teleport_non_finite_entries_raise_value_error(self):
        S_nan = np.array([[1.0, np.nan], [0.0, 0.0]], dtype=float)
        with self.assertRaisesRegex(ValueError, r"non-finite values"):
            compute_publication_rank_teleport(S_nan)

        S_inf = scipy.sparse.csr_matrix(np.array([[1.0, np.inf], [0.0, 0.0]], dtype=float))
        with self.assertRaisesRegex(ValueError, r"non-finite values"):
            compute_publication_rank_teleport(S_inf)

    def test_teleport_rows_must_sum_to_1_or_0_otherwise_error(self):
        # First row sums to 0.5 -> invalid; second row dangling -> OK
        S = scipy.sparse.csr_matrix(np.array([[0.2, 0.3], [0.0, 0.0]], dtype=float))
        with self.assertRaisesRegex(ValueError, r"rows must sum to 1 or 0"):
            compute_publication_rank_teleport(S)

    def test_teleport_vector_validation_errors(self):
        S = scipy.sparse.identity(3, dtype=float, format="csr")

        with self.assertRaisesRegex(ValueError, r"teleport has incompatible size"):
            compute_publication_rank_teleport(S, teleport=np.array([1.0, 0.0], dtype=float))

        with self.assertRaisesRegex(ValueError, r"teleport must be non-negative"):
            compute_publication_rank_teleport(S, teleport=np.array([1.0, -0.5, 0.5], dtype=float))

        with self.assertRaisesRegex(ValueError, r"teleport must sum to a positive value"):
            compute_publication_rank_teleport(S, teleport=np.array([0.0, 0.0, 0.0], dtype=float))

    def test_init_vector_validation_errors(self):
        S = scipy.sparse.identity(3, dtype=float, format="csr")

        with self.assertRaisesRegex(ValueError, r"init has incompatible size"):
            compute_publication_rank_teleport(S, init=np.array([1.0, 0.0], dtype=float))

        with self.assertRaisesRegex(ValueError, r"init must sum to a positive value"):
            compute_publication_rank_teleport(S, init=np.array([0.0, 0.0, 0.0], dtype=float))

    def test_compute_publication_rank_teleport_progress_true(self):
        # 2-node graph with a dangling node (node 1)
        # 0 -> 1; 1 -> (no out-links)
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        S = adjacency_to_stochastic_matrix(adj)

        # Capture any progress output (tqdm or print fallback)
        f_out, f_err = io.StringIO(), io.StringIO()
        with redirect_stdout(f_out), redirect_stderr(f_err):
            r = compute_publication_rank_teleport(
                S,
                alpha=0.85,
                tol=1e-10,
                max_iter=1000,
                progress=True,  # exercise the progress=True path
            )

        # Basic correctness checks
        self.assertEqual(r.shape, (2,))
        self.assertTrue(np.all(r >= -1e-15))  # numerical guard
        self.assertAlmostEqual(float(r.sum()), 1.0, places=12)

        # Ensure we got some output or at least did not crash due to progress handling.
        # Depending on environment (tqdm installed or not), output may be empty or not.
        _ = f_out.getvalue()
        _ = f_err.getvalue()

    def test_callback_invoked_and_early_stop(self):
        # 3-node graph with one dangling node (row 3)
        S = np.array(
            [
                [0.0, 1.0, 0.0],  # node 0 -> node 1
                [0.0, 0.0, 1.0],  # node 1 -> node 2
                [0.0, 0.0, 0.0],  # node 2 dangling
            ],
            dtype=float,
        )

        calls = []
        stop_at_iter = 3

        def cb(it, delta, r):
            # record and validate callback payload
            calls.append((it, float(delta), r.copy()))
            self.assertIsInstance(it, int)
            self.assertGreaterEqual(it, 1)
            self.assertEqual(r.shape, (3,))
            self.assertTrue(np.all(np.isfinite(r)))
            self.assertGreaterEqual(float(delta), 0.0)
            # request early stop at stop_at_iter
            return it >= stop_at_iter

        r = compute_publication_rank_teleport(
            S,
            alpha=0.85,
            tol=0.0,  # ensure we don't stop due to tolerance
            max_iter=100,
            callback=cb,
            progress=False,
        )

        # Callback is invoked twice per iteration in the current implementation
        self.assertEqual(len(calls), stop_at_iter * 2)
        self.assertEqual(calls[-1][0], stop_at_iter)

        # Returned rank is a valid probability distribution
        self.assertEqual(r.shape, (3,))
        self.assertAlmostEqual(float(r.sum()), 1.0, places=12)
        self.assertTrue(np.all(r >= 0.0))


if __name__ == "__main__":
    unittest.main()
