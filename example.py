"""
Comprehensive example: Ranking publications in a citation network using paperank.

This script illustrates the end-to-end workflow for analyzing scholarly impact with the paperank package:
  1. It starts from a single DOI and automatically collects a citation neighborhood,
     including both works that cite the target publication and works cited by it,
     up to a user-defined number of steps in each direction.
  2. It computes PapeRank scores (a PageRank-like metric) for all publications in the neighborhood,
     quantifying their relative importance within the local citation graph.
  3. It saves the ranked results to a file in either JSON or CSV format, including metadata such as
     DOI, rank, score, authors, title, and publication year for each item.

How to use:
  - Set the DOI of interest in `test_doi`.
  - Adjust `forward_steps` and `backward_steps` to control the size and depth of the citation network.
  - Choose the output format ("json" or "csv").
  - Advanced: tune tol/max_iter; `teleport=None` uses uniform teleport.
  - Run the script to generate a file with ranked publication data for the citation neighborhood.

This example is intended for demonstration, validation, and practical testing of the paperank package.
It can be adapted for batch analysis, integration into larger workflows, or as a template for custom
"""

from paperank.paperank_core import crawl_and_rank


def main():
    test_doi = "10.1016/j.ejor.2005.01.053"
    output_format = "json"  # Change to "csv" as needed
    # Note: teleport vector must match the number of nodes; None ⇒ uniform teleport.
    crawl_and_rank(
        doi=test_doi,
        forward_steps=2,
        backward_steps=2,
        alpha=0.85,
        output_format=output_format,
        debug=False,
        progress=True,
        tol=1e-12,
        max_iter=20000,
        teleport=None,
    )


if __name__ == "__main__":
    main()