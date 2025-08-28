from typing import Set, Dict, List, Any, Optional, Union
from .crossref import get_cited_dois
from .open_citations import get_citing_dois
try:
    from tqdm import tqdm  # optional
except Exception:
    tqdm = None

def collect_cited_recursive(
    doi: str,
    depth: int,
    visited: Optional[Set[str]] = None,
    flatten: bool = False,
    max_nodes: Optional[int] = None,
    progress: Optional[bool] = None
) -> Union[Dict[str, List[str]], List[str]]:
    """
    Recursively collect all articles cited by the given DOI up to 'depth' levels.

    Args:
        doi: The DOI of the starting article.
        depth: Maximum recursion depth (N).
        visited: Internal set to avoid duplicate DOIs.
        flatten: If True, return a flat list of DOIs instead of a tree.
        max_nodes: If set, stops recursion after this many unique DOIs.
        progress: If True and tqdm is available, show a progress bar per depth level.

    Returns:
        If flatten is False:
            dict mapping each DOI to its list of cited DOIs.
        If flatten is True:
            list of unique cited DOIs (excluding the root).
    """
    if visited is None:
        visited = set()
    if depth < 1 or doi in visited or (max_nodes is not None and len(visited) >= max_nodes):
        return [] if flatten else {}
    visited.add(doi)
    try:
        cited = get_cited_dois(doi)
        cited_dois = cited.get("cited_dois", []) or []
    except Exception:
        cited_dois = []
    if flatten:
        out: List[str] = []
        seen: Set[str] = set()
        def dfs(node: str, remaining: int):
            if remaining < 1 or node in visited:
                return
            visited.add(node)
            try:
                nxt = get_cited_dois(node).get("cited_dois", []) or []
            except Exception:
                nxt = []
            for x in nxt:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                dfs(x, remaining - 1)
        iterable = cited_dois
        if progress and tqdm is not None:
            iterable = tqdm(cited_dois, desc=f"Depth {depth} citations for {doi}", leave=False)
        for c in iterable:
            if c not in seen:
                seen.add(c)
                out.append(c)
            dfs(c, depth - 1)
        return out
    else:
        result: Dict[str, List[str]] = {doi: cited_dois}
        iterable = cited_dois
        if progress and tqdm is not None:
            iterable = tqdm(cited_dois, desc=f"Depth {depth} citations for {doi}", leave=False)
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
    progress: Optional[bool] = None
) -> Union[Dict[str, List[str]], List[str]]:
    """
    Recursively collect all articles citing the given DOI up to 'depth' levels.

    Args:
        doi: The DOI of the starting article.
        depth: Maximum recursion depth (N).
        visited: Internal set to avoid duplicate DOIs.
        flatten: If True, return a flat list of DOIs instead of a tree.
        max_nodes: If set, stops recursion after this many unique DOIs.
        progress: If True and tqdm is available, show a progress bar per depth level.

    Returns:
        If flatten is False:
            dict mapping each DOI to its list of citing DOIs.
        If flatten is True:
            list of unique citing DOIs (excluding the root).
    """
    if visited is None:
        visited = set()
    if depth < 1 or doi in visited or (max_nodes is not None and len(visited) >= max_nodes):
        return [] if flatten else {}
    visited.add(doi)
    try:
        citing = get_citing_dois(doi)
        citing_dois = citing.get("citing_dois", []) or []
    except Exception:
        citing_dois = []
    if flatten:
        out: List[str] = []
        seen: Set[str] = set()
        def dfs(node: str, remaining: int):
            if remaining < 1 or node in visited:
                return
            visited.add(node)
            try:
                nxt = get_citing_dois(node).get("citing_dois", []) or []
            except Exception:
                nxt = []
            for x in nxt:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                dfs(x, remaining - 1)
        iterable = citing_dois
        if progress and tqdm is not None:
            iterable = tqdm(citing_dois, desc=f"Depth {depth} citing for {doi}", leave=False)
        for c in iterable:
            if c not in seen:
                seen.add(c)
                out.append(c)
            dfs(c, depth - 1)
        return out
    else:
        result: Dict[str, List[str]] = {doi: citing_dois}
        iterable = citing_dois
        if progress and tqdm is not None:
            iterable = tqdm(citing_dois, desc=f"Depth {depth} citing for {doi}", leave=False)
        for c in iterable:
            subtree = collect_citing_recursive(c, depth - 1, visited, flatten=False, max_nodes=max_nodes, progress=progress)
            result.update(subtree)
        return result

def get_citation_neighborhood(
    doi: str,
    forward_steps: int = 1,
    backward_steps: int = 1,
    progress: Optional[bool] = True
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