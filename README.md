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

## Requirements and configuration

- Python 3.8+ is recommended.
- Set CROSSREF_MAILTO to help Crossref identify your traffic and improve reliability:

  macOS/Linux (bash/zsh):
  ```
  export CROSSREF_MAILTO="your.email@example.com"
  ```

- Progress parameter (used across APIs): one of
  - False: no progress
  - True: basic progress (or fallback)
  - 'tqdm': explicitly request tqdm progress bars
  - int: print every N iterations/steps

---

## Quick Start

Here’s a minimal example to rank publications in a citation neighborhood:

```python
from paperank.paperank_core import crawl_and_rank_frontier

# Set your target DOI
doi = "10.1016/j.ejor.2005.01.053"

# Run the analysis
results = crawl_and_rank_frontier(
    doi=doi,
    steps=2,
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

You can fine-tune the crawl and ranking via the following parameters:

- `min_year`: Optional minimum publication year to include during crawling (filters older works).
- `min_citations`: Optional minimum total citation count to include during crawling (filters low-signal works).
- `tol`: Convergence tolerance for the power iteration (default `1e-12`).
- `max_iter`: Maximum number of power-iteration steps (default `10000`).
- `teleport`: Optional teleportation distribution (NumPy array of size N), non-negative and summing to 1. If `None`, a uniform distribution is used.

Example:

```python
from paperank.paperank_core import crawl_and_rank_frontier

results = crawl_and_rank_frontier(
    doi="10.1016/j.ejor.2005.01.053",
    steps=1,
    min_year=2000,       
    min_citations=5,     
    alpha=0.85,
    tol=1e-12,
    max_iter=20000,
    teleport=None,
)
```

---

## Main API

- `crawl_and_rank_frontier`:  
  End-to-end workflow for crawling a citation network and ranking publications.

- `rank`:  
  Compute PapeRank scores for a list of DOIs.

- `rank_and_save_publications_JSON`:  
  Save ranked results to a JSON file.

- `rank_and_save_publications_CSV`:  
  Save ranked results to a CSV file.

- `crawl_citation_neighborhood`:  
  Iteratively crawl 1-hop bidirectional neighborhoods and union results.

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