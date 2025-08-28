import numpy as np
import scipy.sparse
from typing import Optional, Callable, Union, Any

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
        ValueError: If the input matrix is not square.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
        
    mat = adj_matrix.tocsr().astype(np.float64, copy=False)
    row_sums = np.array(mat.sum(axis=1)).flatten()
    for i in range(mat.shape[0]):
        if row_sums[i] > 0:
            mat.data[mat.indptr[i]:mat.indptr[i+1]] /= row_sums[i]
    return mat

def apply_random_jump(
    stochastic_matrix: scipy.sparse.spmatrix,
    alpha: float = 0.85
) -> scipy.sparse.csr_matrix:
    """
    Modify the stochastic matrix to allow a random jump with probability (1 - alpha).
    The probability of following a citation link is alpha.
    The probability of a random jump to any paper is (1 - alpha) / N.

    Dangling rows (all zeros) are replaced with a uniform distribution before teleportation
    so that each row sums to 1 after applying the random jump.

    Args:
        stochastic_matrix: Row-stochastic matrix (scipy.sparse), shape (N, N).
        alpha: Probability of following a citation link (float, 0 <= alpha <= 1).

    Returns:
        scipy.sparse.csr_matrix: Matrix with random jump applied, shape (N, N).
    """
    N = stochastic_matrix.shape[0]
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
    progress: Optional[Union[int, str]] = None
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

    row_sums = np.asarray(S.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError("Input must be row-stochastic (each row sums to 1). Consider apply_random_jump first.")

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
    if progress == 'tqdm':
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_iter, desc="PapeRank", unit="it", leave=False)
        except Exception:
            progress = 10

    for it in range(max_iter):
        r_next = ST @ r
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
        elif isinstance(progress, int) and ((it + 1) % progress == 0 or delta < tol):
            print(f"[PapeRank] iter={it + 1}/{max_iter} delta={delta:.3e}")

        if delta < tol:
            if pbar:
                pbar.close()
            return r_next
        r = r_next

    if pbar:
        pbar.close()
    return r