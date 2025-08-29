# paperank

**A Publication Ranking and Citation Network Analysis Tools**

`paperank` is a Python package for analyzing scholarly impact using citation networks. It provides tools to build citation graphs from DOIs, compute PapeRank (a PageRank-like score), fetch publication metadata, and export ranked results. The package is designed for researchers, bibliometricians, and developers interested in quantifying publication influence within local or global citation networks.

For a discussion on the use of PageRank-like scores beyond the web see [Gleich, 2014](https://arxiv.org/abs/1407.5107).

[Use cases](docs/Use%20cases.md).

---

## Features

- **Citation Graph Construction:**  
  Automatically builds a citation network from a starting DOI, including both cited and citing works, with configurable depth.

- **PapeRank Computation:**  
  Calculates PageRank-like scores for all publications in the network, quantifying their relative importance.

- **Metadata Retrieval:**  
  Fetches publication metadata (authors, title, year, etc.) from Crossref and OpenCitations.

- **Export Ranked Results:**  
  Outputs ranked publication lists to JSON or CSV files, including scores and metadata.

- **Robust HTTP Handling:**  
  Uses retry logic for API requests to handle rate limits and transient errors.

---

## Installation

Install via pip (recommended):

```
pip install paperank
```

Or clone the repository and install locally:

```
git clone https://github.com/gwr3n/paperank.git
cd paperank
pip install .
```

Dependencies are managed via `pyproject.toml` and include:
- `numpy`
- `scipy`
- `requests`
- `tqdm`
- `urllib3`

---

## Quick Start

Here’s a minimal example to rank publications in a citation neighborhood:

```python
from paperank.paperank_core import crawl_and_rank

# Set your target DOI
doi = "10.1016/j.ejor.2005.01.053"

# Run the analysis
results = crawl_and_rank(
    doi=doi,
    forward_steps=2,
    backward_steps=2,
    alpha=0.85,
    output_format="json",  # or "csv"
    debug=False,
    progress=True
)
```

This will:
- Collect the citation neighborhood around the DOI
- Compute PapeRank scores
- Save results to a file (`<DOI>.json` or `<DOI>.csv`)

---

## Advanced Parameters

You can fine-tune the PageRank iteration from `rank` and `crawl_and_rank`:

- `tol`: Convergence tolerance (default `1e-12`).
- `max_iter`: Maximum number of iterations (default `10000`).
- `teleport`: Optional teleportation distribution (numpy array of size N), non-negative and summing to 1. If `None`, a uniform distribution is used.

Example:

```python
results = crawl_and_rank(
    doi=doi,
    forward_steps=1,
    backward_steps=1,
    alpha=0.85,
    tol=1e-12,
    max_iter=20000,
    teleport=None,
)
```

---

## Deprecated Function

- `apply_random_jump` is deprecated. It materializes a dense Google matrix and is intended only for very small graphs.  
  Prefer `compute_publication_rank_teleport`, which applies teleportation during iteration without building a dense matrix.  
  If you already used `apply_random_jump`, pass the result to `compute_publication_rank` (not the teleport variant).

---

## Main API

- `crawl_and_rank`:  
  End-to-end workflow for crawling a citation network and ranking publications.

- `rank`:  
  Compute PapeRank scores for a list of DOIs.

- `rank_and_save_publications_JSON`:  
  Save ranked results to a JSON file.

- `rank_and_save_publications_CSV`:  
  Save ranked results to a CSV file.

- `get_citation_neighborhood`:  
  Collects DOIs in the citation neighborhood of a target publication.

---

## Submodules

- `citation_crawler`:  
  Functions for recursive citation/citing DOI collection.

- `citation_matrix`:  
  Builds sparse adjacency matrices for citation graphs.

- `paperank_matrix`:  
  Matrix utilities for stochastic and PageRank computations.

- `crossref`:  
  Metadata retrieval from Crossref.

- `open_citations`:  
  Citing DOI retrieval from OpenCitations.

- `doi_utils`:  
  DOI normalization and utility functions.

---

## Example

See `example.py` for a comprehensive script demonstrating the workflow (including advanced parameters).

---

## Testing

Unit tests are provided in the `tests` directory. Run with:

```
python -m unittest discover tests
```

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use `paperank` in published work, please cite the repository:

```
@software{rossi2025paperank,
  author = {Roberto Rossi},
  title = {paperank: a publication ranking and citation network analysis tools},
  year = {2025},
  url = {https://github.com/gwr3n/paperank}
}
```

---

## Support & Contributions

- Issues and feature requests: [GitHub Issues](https://github.com/gwr3n/paperank/issues)
- Pull requests welcome!

---

## Project Homepage

[https://github.com/gwr3n/paperank](https://github.com/gwr3n/paperank)