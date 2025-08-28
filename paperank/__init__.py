"""
paperank: a publication ranking and citation network analysis tools.

This package provides functions for:
    - Building citation graphs from DOIs
    - Computing PapeRank (PageRank-like scores)
    - Fetching metadata from Crossref and OpenCitations
    - Exporting ranked results to JSON/CSV

Main API:
    - crawl_and_rank
    - rank
    - rank_and_save_publications_JSON
    - rank_and_save_publications_CSV
    - get_citation_neighborhood

Submodules:
    - citation_crawler
    - citation_matrix
    - paperank_matrix
    - crossref
    - open_citations
    - doi_utils
"""

from .paperank_core import (
    crawl_and_rank,
    rank,
    rank_and_save_publications_JSON,
    rank_and_save_publications_CSV,
)
from .citation_crawler import get_citation_neighborhood