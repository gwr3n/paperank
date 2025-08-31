from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from .crossref import extract_authors_title_year, get_cited_dois, get_work_metadata
from .open_citations import get_citing_dois
from .types import ProgressType

try:
    from tqdm import tqdm as _tqdm  # optional
except Exception:
    _tqdm = None  # type: ignore[assignment]
# A safely-typed tqdm alias: Optional callable (guarded before use)
tqdm_progress: Optional[Callable[..., Iterable[Any]]] = _tqdm


@lru_cache(maxsize=200_000)
def _cached_cited(doi: str) -> tuple:
    try:
        res = get_cited_dois(doi)
        return tuple(res.get("cited_dois", []) or [])
    except Exception:
        return tuple()


@lru_cache(maxsize=200_000)
def _cached_citing(doi: str) -> tuple:
    try:
        res = get_citing_dois(doi)
        return tuple(res.get("citing_dois", []) or [])
    except Exception:
        return tuple()


def _get_cited_list(doi: str) -> List[str]:
    return list(_cached_cited(doi))


def _get_citing_list(doi: str) -> List[str]:
    return list(_cached_citing(doi))


def collect_cited_recursive(
    doi: str,
    depth: int,
    visited: Optional[Set[str]] = None,
    flatten: bool = False,
    max_nodes: Optional[int] = None,
    progress: ProgressType = False,
) -> Union[Dict[str, List[str]], List[str]]:
    """
    Recursively collect all articles cited by the given DOI up to 'depth' levels.

    Args:
        doi: The DOI of the starting article.
        depth: Maximum recursion depth (N).
        visited: Internal set to avoid duplicate DOIs across the entire traversal.
            Note: this deduplication applies across branches, so nodes are only
            visited once overall. The resulting structure is a "frontier" view
            (no repeated nodes) rather than a full tree that repeats the same
            node under multiple parents.
        flatten: If True, return a flat list of DOIs instead of a tree.
        max_nodes: If set, stops recursion after this many unique DOIs.
        progress: If True and tqdm is available, show a progress bar per depth level.

    Returns:
        If flatten is False:
            dict mapping each DOI to its list of cited DOIs (without duplicate revisits).
        If flatten is True:
            list of unique cited DOIs (excluding the root).
    """
    if visited is None:
        visited = set()
    if depth < 1 or doi in visited or (max_nodes is not None and len(visited) >= max_nodes):
        return [] if flatten else {}
    visited.add(doi)

    cited_dois = _get_cited_list(doi)

    if flatten:
        out: List[str] = []
        seen: Set[str] = set()

        def dfs(node: str, remaining: int):
            if remaining < 1 or node in visited:
                return
            if max_nodes is not None and len(visited) >= max_nodes:
                return
            visited.add(node)
            nxt = _get_cited_list(node)
            for x in nxt:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                if max_nodes is not None and len(visited) >= max_nodes:
                    return
                dfs(x, remaining - 1)

        iterable = cited_dois
        if (progress is True or progress == "tqdm") and tqdm_progress is not None:
            iterable = tqdm_progress(cited_dois, desc=f"Depth {depth} citations for {doi}", leave=False)
        for c in iterable:
            if max_nodes is not None and len(visited) >= max_nodes:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
            dfs(c, depth - 1)
        return out
    else:
        result: Dict[str, List[str]] = {doi: cited_dois}
        iterable = cited_dois
        if (progress is True or progress == "tqdm") and tqdm_progress is not None:
            iterable = tqdm_progress(cited_dois, desc=f"Depth {depth} citations for {doi}", leave=False)
        for c in iterable:
            subtree = collect_cited_recursive(c, depth - 1, visited, flatten=False, max_nodes=max_nodes, progress=progress)
            result.update(subtree)
        return result


def collect_citing_recursive(
    doi: str,
    depth: int,
    visited: Optional[Set[str]] = None,
    flatten: bool = False,
    max_nodes: Optional[int] = None,
    progress: ProgressType = False,
) -> Union[Dict[str, List[str]], List[str]]:
    """
    Recursively collect all articles citing the given DOI up to 'depth' levels.

    Args:
        doi: The DOI of the starting article.
        depth: Maximum recursion depth (N).
        visited: Internal set to avoid duplicate DOIs across the entire traversal.
            Note: this deduplication applies across branches, so nodes are only
            visited once overall. The resulting structure is a "frontier" view
            (no repeated nodes) rather than a full tree that repeats the same
            node under multiple parents.
        flatten: If True, return a flat list of DOIs instead of a tree.
        max_nodes: If set, stops recursion after this many unique DOIs.
        progress: If True and tqdm is available, show a progress bar per depth level.

    Returns:
        If flatten is False:
            dict mapping each DOI to its list of citing DOIs (without duplicate revisits).
        If flatten is True:
            list of unique citing DOIs (excluding the root).
    """
    if visited is None:
        visited = set()
    if depth < 1 or doi in visited or (max_nodes is not None and len(visited) >= max_nodes):
        return [] if flatten else {}
    visited.add(doi)

    citing_dois = _get_citing_list(doi)

    if flatten:
        out: List[str] = []
        seen: Set[str] = set()

        def dfs(node: str, remaining: int):
            if remaining < 1 or node in visited:
                return
            if max_nodes is not None and len(visited) >= max_nodes:
                return
            visited.add(node)
            nxt = _get_citing_list(node)
            for x in nxt:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                if max_nodes is not None and len(visited) >= max_nodes:
                    return
                dfs(x, remaining - 1)

        iterable = citing_dois
        if (progress is True or progress == "tqdm") and tqdm_progress is not None:
            iterable = tqdm_progress(citing_dois, desc=f"Depth {depth} citing for {doi}", leave=False)
        for c in iterable:
            if max_nodes is not None and len(visited) >= max_nodes:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
            dfs(c, depth - 1)
        return out
    else:
        result: Dict[str, List[str]] = {doi: citing_dois}
        iterable = citing_dois
        if (progress is True or progress == "tqdm") and tqdm_progress is not None:
            iterable = tqdm_progress(citing_dois, desc=f"Depth {depth} citing for {doi}", leave=False)
        for c in iterable:
            subtree = collect_citing_recursive(c, depth - 1, visited, flatten=False, max_nodes=max_nodes, progress=progress)
            result.update(subtree)
        return result


def get_citation_neighborhood(
    doi: str, forward_steps: int = 1, backward_steps: int = 1, progress: ProgressType = True
) -> List[str]:
    """
    Given a DOI, return a flat list containing:
      - all citing DOIs (up to 'forward_steps' recursive steps forward)
      - the original DOI
      - all cited DOIs (up to 'backward_steps' recursive steps back)

    Args:
        doi: The DOI to start from.
        forward_steps: Number of steps to follow citing links.
        backward_steps: Number of steps to follow cited links.
        progress: If True and tqdm is available, show progress bars during fetching.

    Returns:
        List of unique DOIs, including the original, citing, and cited DOIs.
        Order is preserved and duplicates are removed.
    """
    citing = collect_citing_recursive(doi, depth=forward_steps, flatten=True, progress=progress)
    cited = collect_cited_recursive(doi, depth=backward_steps, flatten=True, progress=progress)
    result = [doi] + [d for d in citing if d != doi] + [d for d in cited if d != doi]
    result = list(dict.fromkeys(result))  # Deduplicate, preserve order
    return result


def crawl_citation_neighborhood(
    doi: List[str],
    steps: int = 1,
    min_year: Optional[int] = None,
    min_citations: Optional[int] = None,
    progress: ProgressType = True,
) -> List[str]:
    """
    Crawl citation neighborhoods iteratively.

    Behavior:
    - steps == 1:
      * if doi has one element: identical to get_citation_neighborhood(doi[0], 1, 1)
      * if doi has multiple elements: union of get_citation_neighborhood(d, 1, 1) for all d
    - steps > 1:
      * Let S0 = crawl_citation_neighborhood(doi, 1)
        S1 = crawl_citation_neighborhood(S0, 1)
        ...
        S_{steps-1} similarly;
        Return the union (deduplicated) of S0..S_{steps-1}, preserving order.

    Args:
        doi: List of starting DOIs.
        steps: Number of iterative crawling steps.
        progress: If True and tqdm is available, show progress bars during fetching.

    Returns:
        List of unique DOIs in the crawled neighborhood.

    Note: if steps < 1, returns an empty list.
    """
    if steps < 1:
        return []

    # Compute the union (deduped, order-preserving) of 1-hop neighborhoods
    def one_hop(seeds: List[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        iterable = seeds
        if (progress is True or progress == "tqdm") and tqdm_progress is not None:
            iterable = tqdm_progress(seeds, desc="Crawling seeds", leave=False)
        for d in iterable:
            lst = get_citation_neighborhood(d, 1, 1, progress=False)
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    out.append(x)

        # Apply independent filters: min_year and min_citations
        if min_year is not None or min_citations is not None:
            filtered: List[str] = []
            iterable_filter = out
            if (progress is True or progress == "tqdm") and tqdm_progress is not None:
                iterable_filter = tqdm_progress(out, desc="Filtering DOIs", leave=False)
            for doi_item in iterable_filter:
                drop = False

                # Filter by publication year (keep only year >= min_year if known)
                if min_year is not None:
                    try:
                        meta = get_work_metadata(doi_item) or {}
                    except Exception:
                        meta = {}
                    # Reuse shared extractor which accepts envelope or message
                    _, _, year_val = extract_authors_title_year(meta)
                    if year_val is not None and year_val < min_year:
                        drop = True

                # Filter by citation count (keep only >= min_citations)
                if not drop and min_citations is not None:
                    try:
                        citations_count = len(_get_citing_list(doi_item))
                    except Exception:
                        citations_count = 0
                    if citations_count < min_citations:
                        drop = True

                if not drop:
                    filtered.append(doi_item)
            out = filtered

        return out

    # Step 0
    seeds = list(dict.fromkeys(doi))  # dedupe input seeds, preserve order
    step_lists: List[List[str]] = []
    current = one_hop(seeds)
    step_lists.append(current)

    # Steps 1..(steps-1)
    for _ in range(1, steps):
        current = one_hop(current)
        step_lists.append(current)

    # Union across all step lists, preserving first-seen order
    result: List[str] = []
    seen_total: Set[str] = set()
    for lst in step_lists:
        for x in lst:
            if x not in seen_total:
                seen_total.add(x)
                result.append(x)
    return result


# New: cache management
def clear_caches() -> None:
    """Clear LRU caches used by citation crawler helpers."""
    try:
        _cached_cited.cache_clear()
    except Exception:
        pass
    try:
        _cached_citing.cache_clear()
    except Exception:
        pass
