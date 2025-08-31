"""
Comprehensive example: Ranking publications in a citation network using paperank
with the frontier-based crawler (crawl_and_rank_frontier).

This script illustrates an end-to-end workflow:
  1. Starting from one or more seed DOIs, it iteratively expands a symmetric
     1-hop frontier (both citations and references) per step. After N steps,
     the neighborhood contains works within N hops of the seeds.
  2. It computes PapeRank scores (PageRank-like) over the local citation graph.
  3. It saves ranked results as JSON or CSV with DOI, rank, score, authors, title, year.

How to use:
  - Set the seed DOI in `test_doi` (or pass a list of DOIs to seed multiple topics).
  - Adjust `steps` to control neighborhood radius (each step = one bidirectional hop).
  - Optional filters: `min_year` and `min_citations` to constrain crawl scope.
  - Choose the output format ("json" or "csv").
  - Advanced: tune tol/max_iter; `teleport=None` uses uniform teleport.
  - Run the script to generate ranked publication data for the crawled neighborhood.

This example supersedes the deprecated bidirectional_neighborhood API and reflects
the recommended crawl_and_rank_frontier-based workflow.
"""

from paperank.paperank_core import crawl_and_rank_frontier

def main_crawl_and_rank_frontier():
    test_doi = "10.1016/j.ejor.2005.01.053"
    output_format = "json"  # Change to "csv" as needed
    # Note: teleport vector must match the number of nodes; None ⇒ uniform teleport.
    crawl_and_rank_frontier(
        doi=test_doi,
        steps=2,
        min_year=2000,
        min_citations=5,
        alpha=0.85,
        output_format=output_format,
        debug=False,
        progress=True,
        tol=1e-12,
        max_iter=20000,
        teleport=None,
    )

if __name__ == "__main__":
    main_crawl_and_rank_frontier()