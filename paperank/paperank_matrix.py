import numpy as np
import scipy.sparse
import warnings
from typing import Optional, Callable, Union, Any, Literal

from .types import ProgressType

def adjacency_to_stochastic_matrix(
    adj_matrix: Union[np.ndarray, scipy.sparse.spmatrix]
) -> scipy.sparse.csr_matrix:
    """
    Convert an adjacency matrix to a row-stochastic matrix (CSR format).
    Each row sums to 1. If a row is all zeros, it remains zeros.

    Args:
        adj_matrix: Square adjacency matrix (numpy or scipy.sparse), shape (N, N).

    Returns:
        scipy.sparse.csr_matrix: Row-stochastic matrix of shape (N, N).

    Raises:
        ValueError: If the input matrix is not square or contains invalid values.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Ensure CSR float64 and DO NOT mutate caller: force copy
    mat = scipy.sparse.csr_matrix(adj_matrix, dtype=np.float64, copy=True)

    # Validate values: finite and non-negative
    data = mat.data
    if data.size:
        if not np.all(np.isfinite(data)):
            raise ValueError("Adjacency matrix contains non-finite values (NaN/Inf).")
        if np.any(data < 0):
            raise ValueError("Adjacency matrix must be non-negative.")

    # Vectorized row normalization: multiply by inverse row sums per row
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    inv_row_sums = np.zeros_like(row_sums, dtype=np.float64)
    nz = row_sums > 0
    inv_row_sums[nz] = 1.0 / row_sums[nz]
    # Broadcasting across columns
    mat = mat.multiply(inv_row_sums[:, np.newaxis])
    return mat.tocsr()

def apply_random_jump(
    stochastic_matrix: scipy.sparse.spmatrix,
    alpha: float = 0.85
) -> scipy.sparse.csr_matrix:
    """
    DEPRECATED: Use compute_publication_rank_teleport(...) instead.

    Modify the stochastic matrix to allow a random jump with probability (1 - alpha).
    The probability of following a citation link is alpha.
    The probability of a random jump to any paper is (1 - alpha) / N.

    Dangling rows (all zeros) are replaced with a uniform distribution before teleportation
    so that each row sums to 1 after applying the random jump.

    Important:
    - This function materializes the dense Google matrix (N x N), which is O(N^2) memory.
    - Intended for small graphs only. For large graphs, use
      compute_publication_rank_teleport(...) which applies teleportation during iteration
      without building a dense matrix.
    - The returned matrix already includes teleportation; do NOT pass it to
      compute_publication_rank_teleport again (use compute_publication_rank instead).

    Args:
        stochastic_matrix: Row-stochastic matrix (scipy.sparse), shape (N, N).
        alpha: Probability of following a citation link (float, 0 <= alpha <= 1).

    Returns:
        scipy.sparse.csr_matrix: Dense matrix with random jump applied, shape (N, N).
    """
    warnings.warn(
        (
            "apply_random_jump is deprecated; use compute_publication_rank_teleport instead. "
            "It also materializes a dense N×N matrix and is intended only for small graphs."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    N = stochastic_matrix.shape[0]
    if N == 0:
        return scipy.sparse.csr_matrix((0, 0))

    S = stochastic_matrix.toarray()

    row_sums = S.sum(axis=1)
    zero_rows = (row_sums == 0)
    if np.any(zero_rows):
        S[zero_rows, :] = 1.0 / N

    S = alpha * S + (1 - alpha) * (1.0 / N)
    return scipy.sparse.csr_matrix(S)

def compute_publication_rank(
    stochastic_matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    tol: float = 1e-10,
    max_iter: int = 1000,
    init: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, float, np.ndarray], Any]] = None,
    progress: ProgressType = False
) -> np.ndarray:
    """
    Compute the stationary distribution (PapeRank) for a row-stochastic matrix S.
    Finds r such that r = r S, with r being a probability vector.

    Args:
        stochastic_matrix: Row-stochastic matrix (numpy or scipy.sparse), shape (N, N).
        tol: L1 tolerance for convergence (float).
        max_iter: Maximum number of iterations (int).
        init: Optional initial probability vector (np.ndarray, shape (N,)), defaults to uniform.
        callback: Optional callable(iteration, delta, r) -> bool|None.
            If it returns True, iteration stops early.
        progress: None for no output; int N to print every N iterations;
            or 'tqdm' to show a progress bar (requires tqdm).

    Returns:
        np.ndarray: Rank vector r of shape (N,), non-negative and sums to 1.

    Raises:
        ValueError: If input matrix is not square, not row-stochastic, or init is invalid.
    """
    S = scipy.sparse.csr_matrix(stochastic_matrix, dtype=np.float64)
    n = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValueError("stochastic_matrix must be square")
    if n == 0:
        return np.asarray([], dtype=np.float64)

    row_sums = np.asarray(S.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError(
            "Input must be row-stochastic (each row sums to 1). "
            "If there are dangling rows (sum=0), use compute_publication_rank_teleport."
        )

    if init is None:
        r = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        r = np.asarray(init, dtype=np.float64).ravel()
        if r.size != n:
            raise ValueError("init has incompatible size")
        s = r.sum()
        if s <= 0:
            raise ValueError("init must sum to a positive value")
        if np.any(r < 0):
            raise ValueError("init must be non-negative")
        r /= s

    ST = S.transpose().tocsr()

    pbar = None
    if progress == 'tqdm' or progress is True:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_iter, desc="PapeRank", unit="it", leave=False)
        except Exception:
            # fallback to printing every 10 if tqdm unavailable but True requested
            if progress is True:
                progress = 10

    for it in range(max_iter):
        r_next = ST @ r
        s = r_next.sum()
        if s <= 0:
            if pbar:
                pbar.close()
            raise ValueError("Encountered non-positive total probability during iteration")
        r_next /= s
        if np.any(r_next < -1e-15):
            if pbar:
                pbar.close()
            raise ValueError("Encountered negative probability during iteration; ensure S is non-negative row-stochastic.")

        delta = np.linalg.norm(r_next - r, 1)

        if callback is not None:
            should_stop = bool(callback(it + 1, delta, r_next))
            if should_stop:
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"delta={delta:.3e} (stopped)")
                    pbar.close()
                return r_next

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(f"delta={delta:.3e}")
        elif (isinstance(progress, int) and not isinstance(progress, bool) and progress > 0
              and ((it + 1) % progress == 0 or delta < tol)):
            print(f"[PapeRank] iter={it + 1}/{max_iter} delta={delta:.3e}")

        if delta < tol:
            if pbar:
                pbar.close()
            return r_next
        r = r_next

    if pbar:
        pbar.close()
    return r

def compute_publication_rank_teleport(
    stochastic_matrix: Union[np.ndarray, scipy.sparse.spmatrix],
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 1000,
    init: Optional[np.ndarray] = None,
    teleport: Optional[np.ndarray] = None,
    callback: Optional[Callable[[int, float, np.ndarray], Any]] = None,
    progress: ProgressType = False,
) -> np.ndarray:
    """
    Compute PapeRank via power-iteration with teleportation, without materializing
    the dense Google matrix. Handles dangling rows efficiently.

    r_{t+1} = alpha * r_t * S
               + alpha * (sum_{i in dangling} r_t[i]) * v
               + (1 - alpha) * v

    where v is the teleportation distribution (defaults to uniform).

    Args:
        stochastic_matrix: Row-stochastic matrix (scipy.sparse/numpy), shape (N, N). Rows may be all-zero (dangling).
        alpha: Probability of following a citation link (0 <= alpha <= 1).
        tol: L1 tolerance for convergence.
        max_iter: Maximum iterations.
        init: Optional initial distribution (size N). Will be normalized.
        teleport: Optional teleportation distribution v (size N), non-negative and sums to 1. Defaults to uniform.
        callback: Optional callable(iteration, delta, r)->bool to stop early when returns True.
        progress: None for no output; integer N to print every N iterations; or 'tqdm' to show a progress bar.

    Returns:
        Stationary distribution r (size N) summing to 1.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    S = scipy.sparse.csr_matrix(stochastic_matrix, dtype=np.float64)
    n = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValueError("stochastic_matrix must be square")
    if n == 0:
        return np.asarray([], dtype=np.float64)

    row_sums = np.asarray(S.sum(axis=1)).ravel()
    # Validate rows are either stochastic (~1) or dangling (~0)
    valid_rows = np.isclose(row_sums, 1.0, atol=1e-9) | np.isclose(row_sums, 0.0, atol=1e-12)
    if not np.all(valid_rows):
        raise ValueError("Input must be row-stochastic; rows must sum to 1 or 0 (dangling). Consider adjacency_to_stochastic_matrix.")
    dangling = row_sums == 0.0

    if teleport is None:
        v = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        v = np.asarray(teleport, dtype=np.float64).ravel()
        if v.size != n:
            raise ValueError("teleport has incompatible size")
        if np.any(v < 0):
            raise ValueError("teleport must be non-negative")
        s = v.sum()
        if s <= 0:
            raise ValueError("teleport must sum to a positive value")
        v /= s

    if init is None:
        r = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        r = np.asarray(init, dtype=np.float64).ravel()
        if r.size != n:
            raise ValueError("init has incompatible size")
        s = r.sum()
        if s <= 0:
            raise ValueError("init must sum to a positive value")
        r /= s

    ST = S.transpose().tocsr()

    pbar = None
    if progress == 'tqdm' or progress is True:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_iter, desc="PapeRank", unit="it", leave=False)
        except Exception:
            if progress is True:
                progress = 10

    for it in range(max_iter):
        r_link = ST @ r  # equals r @ S
        d_mass = float(r[dangling].sum()) if dangling.any() else 0.0
        r_next = alpha * r_link + alpha * d_mass * v + (1.0 - alpha) * v
        # Numerical guard: normalize
        s = r_next.sum()
        if s <= 0:
            if pbar:
                pbar.close()
            raise ValueError("Encountered non-positive total probability during iteration")
        r_next /= s

        delta = np.linalg.norm(r_next - r, 1)

        if callback is not None:
            should_stop = bool(callback(it + 1, delta, r_next))
            if should_stop:
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"delta={delta:.3e} (stopped)")
                    pbar.close()
                return r_next

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(f"delta={delta:.3e}")
        elif (isinstance(progress, int) and not isinstance(progress, bool) and progress > 0
              and ((it + 1) % progress == 0 or delta < tol)):
            print(f"[PapeRank] iter={it + 1}/{max_iter} delta={delta:.3e}")

        if delta < tol:
            if pbar:
                pbar.close()
            return r_next
        r = r_next

    if pbar:
        pbar.close()
    return r