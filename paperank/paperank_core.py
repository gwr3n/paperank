import numpy as np
import json
import csv
import warnings
from typing import List, Tuple, Dict, Any, Optional, Union, Literal

from .citation_crawler import get_citation_neighborhood, crawl_citation_neighborhood
from .citation_matrix import build_citation_sparse_matrix
from .paperank_matrix import adjacency_to_stochastic_matrix, compute_publication_rank_teleport
from .crossref import get_work_metadata, extract_authors_title_year
from .types import ProgressType

def crawl_and_rank_bidirectional_neighborhood(
    doi: str,
    forward_steps: int = 1,
    backward_steps: int = 1,
    alpha: float = 0.85,
    output_format: str = "json",
    debug: bool = False,
    progress: ProgressType = True,
    tol: float = 1e-12,
    max_iter: int = 10000,
    teleport: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """
    Collect a citation neighborhood for a given DOI, compute PapeRank scores,
    and save the ranked results to a file in the specified format.

    Deprecated:
        This function is deprecated and will be removed in a future release.
        Use crawl_and_rank_frontier(doi, steps=1, ...) for a 1-hop bidirectional
        neighborhood. For deeper traversals, use steps=max(forward_steps, backward_steps).
        Note: asymmetric forward/backward depths are not supported by the replacement.

    Args:
        doi: The DOI of the publication to analyze.
        forward_steps: Number of steps to follow citing links (forward).
        backward_steps: Number of steps to follow cited links (backward).
        alpha: Probability of following a citation link (PageRank damping factor).
        output_format: Output file format ("json" or "csv").
        debug: If True, prints progress/debug information.
        progress: If True, shows progress bar; if False, disables; if 'tqdm', uses tqdm.
        tol: Convergence tolerance for power iteration.
        max_iter: Maximum number of power-iteration steps.
        teleport: Optional teleportation distribution (size N), non-negative and sums to 1.

    Returns:
        List of tuples (doi, score), sorted by score descending.

    Side Effects:
        Writes a JSON or CSV file with ranked publication data.
    """
    warnings.warn(
        "crawl_and_rank_bidirectional_neighborhood is deprecated and will be removed in a future release. "
        "Use crawl_and_rank_frontier(doi, steps=1, ...) for 1-hop bidirectional neighborhoods, or "
        "steps=max(forward_steps, backward_steps) for deeper crawls (asymmetric depths not supported).",
        DeprecationWarning,
        stacklevel=2,
    )
    doi_filename: str = doi.replace("/", "_").replace(".", "_")
    doi_list: List[str] = get_citation_neighborhood(doi, forward_steps=forward_steps, backward_steps=backward_steps)

    if output_format == "json":
        rank_and_save_publications_JSON(
            doi_list,
            out_path=doi_filename + ".json",
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            teleport=teleport,
            progress=progress,  # pass through for parity
        )
    elif output_format == "csv":
        rank_and_save_publications_CSV(
            doi_list,
            out_path=doi_filename + ".csv",
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            teleport=teleport,
            progress=progress,  # pass through for parity
        )
    else:
        print(f"Unknown output format: {output_format}")

    return rank(doi_list, alpha=alpha, debug=debug, progress=progress, tol=tol, max_iter=max_iter, teleport=teleport)

def crawl_and_rank_frontier(
    doi: Union[str, List[str]],
    steps: int = 1,
    min_year: Optional[int] = None,
    min_citations: Optional[int] = None,
    alpha: float = 0.85,
    output_format: str = "json",
    debug: bool = False,
    progress: ProgressType = True,
    tol: float = 1e-12,
    max_iter: int = 10000,
    teleport: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """
    Crawl citation neighborhoods using crawl_citation_neighborhood and rank the results.

    Args:
        doi: A single DOI (str) or a list of DOIs to use as seeds.
        steps: Number of iterative crawl steps (each step uses 1-hop neighborhoods).
        alpha: PageRank damping factor.
        output_format: "json" or "csv".
        debug: If True, prints progress/debug information.
        progress: Progress behavior; True/False or 'tqdm'.
        tol: Convergence tolerance for power iteration.
        max_iter: Max number of power-iteration steps.
        teleport: Optional teleportation distribution.

    Returns:
        List of tuples (doi, score), sorted by score descending.
    """
    # Normalize seeds
    seeds: List[str] = [doi] if isinstance(doi, str) else list(dict.fromkeys(doi))

    # Crawl with iterative union neighborhoods
    doi_list: List[str] = crawl_citation_neighborhood(seeds, steps=steps, min_year=min_year, min_citations=min_citations, progress=progress)

    # Choose output file base name
    if isinstance(doi, str):
        base_name: str = doi.replace("/", "_").replace(".", "_")
    else:
        base_name = f"crawl_{len(seeds)}_seeds"

    # Save results
    if output_format == "json":
        rank_and_save_publications_JSON(doi_list, out_path=base_name + ".json", alpha=alpha, tol=tol, max_iter=max_iter, teleport=teleport, progress=progress)
    elif output_format == "csv":
        rank_and_save_publications_CSV(doi_list, out_path=base_name + ".csv", alpha=alpha, tol=tol, max_iter=max_iter, teleport=teleport, progress=progress)
    else:
        print(f"Unknown output format: {output_format}")

    # Rank and return
    progress_val = 'tqdm' if debug else progress
    return rank(doi_list, alpha=alpha, debug=debug, progress=progress_val, tol=tol, max_iter=max_iter, teleport=teleport)

def rank(
    doi_list: List[str],
    alpha: float = 0.85,
    debug: bool = False,
    progress: ProgressType = True,
    tol: float = 1e-12,
    max_iter: int = 10000,
    teleport: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """
    Compute the PapeRank for a list of DOIs.

    Args:
        doi_list: List of DOIs to rank.
        alpha: Probability of following a citation link (PageRank damping factor).
        debug: If True, prints progress/debug information.
        progress: If True, shows progress bar; if False, disables; if 'tqdm', uses tqdm.
        tol: Convergence tolerance for power iteration.
        max_iter: Maximum number of power-iteration steps.
        teleport: Optional teleportation distribution (size N), non-negative and sums to 1.

    Returns:
        List of tuples (doi, score), sorted by score descending.
    """
    if debug:
        print("Building citation adjacency matrix...")
    adjacency_matrix, doi_to_idx = build_citation_sparse_matrix(doi_list, progress=progress)
    if debug:
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}, nnz={adjacency_matrix.nnz}")

    if debug:
        print("Computing stochastic matrix (sparse)...")
    S = adjacency_to_stochastic_matrix(adjacency_matrix)
    if debug:
        print(f"Stochastic matrix shape: {S.shape}, nnz={S.nnz}")

    if debug:
        print("Computing PapeRank via sparse power-iteration with teleportation...")
    progress_val = 'tqdm' if debug else progress
    r = compute_publication_rank_teleport(S, alpha=alpha, tol=tol, max_iter=max_iter, teleport=teleport, progress=progress_val)
    if debug:
        print("PapeRank computation completed.")

    idx_to_doi: Dict[int, str] = {idx: doi for doi, idx in doi_to_idx.items()}
    ranked: List[Tuple[str, float]] = sorted(
        ((idx_to_doi[i], float(score)) for i, score in enumerate(r)),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked

def rank_and_save_publications_JSON(
    doi_list: List[str],
    out_path: str,
    alpha: float = 0.85,
    max_results: int = 10,
    progress: ProgressType = True,
    tol: float = 1e-12,
    max_iter: int = 10000,
    teleport: Optional[np.ndarray] = None,
) -> None:
    """
    Rank the given DOIs and save the top results to a JSON file.
    Each entry includes: doi, rank (1-based), score, authors (list), title, year.

    Args:
        doi_list: List of DOIs to rank.
        out_path: Path to output JSON file.
        alpha: Probability of following a citation link (PageRank damping factor).
        max_results: Maximum number of top results to save.
        progress: If True, shows progress bar; if False, disables; if 'tqdm', uses tqdm.
        tol: Convergence tolerance for power iteration.
        max_iter: Maximum number of power-iteration steps.
        teleport: Optional teleportation distribution (size N), non-negative and sums to 1.

    Returns:
        None

    Side Effects:
        Writes a JSON file with ranked publication data.
    """
    ranked: List[Tuple[str, float]] = rank(doi_list, alpha=alpha, progress=progress, tol=tol, max_iter=max_iter, teleport=teleport)

    results: List[Dict[str, Any]] = []
    for rank_idx, (doi, score) in enumerate(ranked[:max_results], start=1):
        try:
            meta: Dict[str, Any] = get_work_metadata(doi) or {}
        except Exception:
            meta = {}
        authors, title, year = extract_authors_title_year(meta)
        results.append({
            "doi": doi,
            "rank": rank_idx,
            "score": score,
            "authors": authors,
            "title": title,
            "year": year,
        })

    payload: Dict[str, Any] = {
        "alpha": alpha,
        "max_results": max_results,
        "total_ranked": len(ranked),
        "items": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def rank_and_save_publications_CSV(
    doi_list: List[str],
    out_path: str,
    alpha: float = 0.85,
    max_results: int = 10,
    progress: ProgressType = True,
    tol: float = 1e-12,
    max_iter: int = 10000,
    teleport: Optional[np.ndarray] = None,
) -> None:
    """
    Rank the given DOIs and save the top results to a CSV file.
    Each entry includes: rank, doi, score, authors, title, year.

    Args:
        doi_list: List of DOIs to rank.
        out_path: Path to output CSV file.
        alpha: Probability of following a citation link (PageRank damping factor).
        max_results: Maximum number of top results to save.
        progress: If True, shows progress bar; if False, disables; if 'tqdm', uses tqdm.
        tol: Convergence tolerance for power iteration.
        max_iter: Maximum number of power-iteration steps.
        teleport: Optional teleportation distribution (size N), non-negative and sums to 1.

    Returns:
        None

    Side Effects:
        Writes a CSV file with ranked publication data.
    """
    ranked: List[Tuple[str, float]] = rank(doi_list, alpha=alpha, progress=progress, tol=tol, max_iter=max_iter, teleport=teleport)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["rank", "doi", "score", "authors", "title", "year"])
        for rank_idx, (doi, score) in enumerate(ranked[:max_results], start=1):
            try:
                meta: Dict[str, Any] = get_work_metadata(doi) or {}
            except Exception:
                meta = {}
            authors, title, year = extract_authors_title_year(meta)
            authors_str: str = "; ".join(authors)
            year_field: Any = year if year is not None else ""
            writer.writerow([rank_idx, doi, score, authors_str, title, year_field])