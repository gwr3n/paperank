import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Callable, Any, Optional, Union, Literal

from .crossref import get_cited_dois
from .open_citations import get_citing_dois
from .doi_utils import normalize_doi
from .types import ProgressType

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm  # optional
except Exception:
    tqdm = None

@lru_cache(maxsize=200_000)
def _cached_cited(doi: str) -> Tuple[str, ...]:
    """
    Cached wrapper for get_cited_dois. Returns tuple of cited DOIs for a given DOI.

    Args:
        doi: DOI string.

    Returns:
        Tuple of cited DOIs (normalized strings).
    """
    try:
        res = get_cited_dois(doi)
        return tuple(res.get("cited_dois", []) or [])
    except Exception:
        return tuple()

@lru_cache(maxsize=200_000)
def _cached_citing(doi: str) -> Tuple[str, ...]:
    """
    Cached wrapper for get_citing_dois. Returns tuple of citing DOIs for a given DOI.

    Args:
        doi: DOI string.

    Returns:
        Tuple of citing DOIs (normalized strings).
    """
    try:
        res = get_citing_dois(doi)
        return tuple(res.get("citing_dois", []) or [])
    except Exception:
        return tuple()

def build_citation_sparse_matrix(
    doi_list: List[str],
    include_citing: bool = False,
    max_workers: Optional[int] = None,
    use_cache: bool = True,
    progress: ProgressType = False
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """
    Build a sparse adjacency matrix representing the citation graph for a list of DOIs.
    Each row i and column j corresponds to doi_list[i] and doi_list[j].
    Entry (i, j) is 1 if doi_list[i] cites doi_list[j], 0 otherwise.

    Optionally, if include_citing is True, also fill edges from items that cite a DOI in the list.

    Args:
        doi_list: List of DOIs to include in the matrix.
        include_citing: If True, also fill edges from items that cite a DOI in the list.
        max_workers: If provided, fetch metadata concurrently with this many threads.
        use_cache: If True, memoize citation/citing lookups in-process (LRU).
        progress: If True or 'tqdm' and tqdm is available, show a progress bar.

    Returns:
        matrix: scipy.sparse.csr_matrix, shape (len(doi_list), len(doi_list)), adjacency matrix.
        doi_to_idx: dict mapping DOI string to matrix index.

    Example:
        >>> matrix, doi_to_idx = build_citation_sparse_matrix(["10.1016/j.ejor.2016.12.001", "10.1080/1540496x.2019.1696189"])
        >>> matrix.shape
        (2, 2)
    """
    # Normalize and deduplicate inputs (preserve order)
    seen = set()
    normed: List[str] = []
    for d in doi_list:
        nd = normalize_doi(d)
        if nd and nd not in seen:
            seen.add(nd)
            normed.append(nd)
    doi_list = normed

    n = len(doi_list)
    doi_to_idx: Dict[str, int] = {doi: idx for idx, doi in enumerate(doi_list)}
    # Choose fetch functions (cached or direct)
    fetch_cited: Callable[[str], Tuple[str, ...]] = _cached_cited if use_cache else (lambda d: tuple((get_cited_dois(d).get("cited_dois", []) or [])))
    fetch_citing: Callable[[str], Tuple[str, ...]] = _cached_citing if use_cache else (lambda d: tuple((get_citing_dois(d).get("citing_dois", []) or [])))

    # Fetch all neighbor lists (serial or parallel)
    def _job(doi: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        cited = fetch_cited(doi)
        citing = fetch_citing(doi) if include_citing else tuple()
        return doi, cited, citing

    results: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
    if max_workers and max_workers > 1:
        workers = min(max_workers, max(1, len(doi_list)))
        pbar = None
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_job, d): d for d in doi_list}
                if (progress is True or progress == 'tqdm') and tqdm is not None:
                    pbar = tqdm(total=len(futures), desc="Fetching citations", unit="doi", leave=False)
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception:
                        # On failure, keep empty lists
                        d = futures[fut]
                        results.append((d, tuple(), tuple()))
                    if pbar:
                        pbar.update(1)
        finally:
            if pbar:
                pbar.close()
    else:
        iterable = doi_list
        if (progress is True or progress == 'tqdm') and tqdm is not None:
            iterable = tqdm(doi_list, desc="Fetching citations", unit="doi", leave=False)
        for d in iterable:
            try:
                results.append(_job(d))
            except Exception:
                results.append((d, tuple(), tuple()))

    # Build edge list (deduplicated) and construct sparse matrix in one shot
    edges: set = set()
    for doi, cited, citing in results:
        i = doi_to_idx.get(doi)
        if i is None:
            continue
        # cited edges: i -> j
        for cd in cited:
            j = doi_to_idx.get(normalize_doi(cd))
            if j is not None:
                edges.add((i, j))
        if include_citing:
            # citing edges: j -> i
            for cg in citing:
                j = doi_to_idx.get(normalize_doi(cg))
                if j is not None:
                    edges.add((j, i))

    if not edges:
        return sp.csr_matrix((n, n), dtype=np.int8), doi_to_idx

    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.int8)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8).tocsr()
    return matrix, doi_to_idx