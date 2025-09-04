# paperank

Core package badges:

![Codecov (with branch)](https://img.shields.io/codecov/c/gh/gwr3n/paperank/main)
 ![Python package](https://img.shields.io/github/actions/workflow/status/gwr3n/paperank/.github%2Fworkflows%2Fpython-package.yml) ![Lint and type-check](https://img.shields.io/github/actions/workflow/status/gwr3n/paperank/.github%2Fworkflows%2Flint-type.yml?branch=main&label=lint%20%2B%20type-check) [![PyPI](https://img.shields.io/pypi/v/paperank)](https://pypi.org/project/paperank/) [![Python versions](https://img.shields.io/pypi/pyversions/paperank)](https://pypi.org/project/paperank/) [![License](https://img.shields.io/github/license/gwr3n/paperank)](LICENSE) [![Downloads](https://static.pepy.tech/badge/paperank)](https://pepy.tech/project/paperank) [![Release](https://img.shields.io/github/v/release/gwr3n/paperank)](https://github.com/gwr3n/paperank/releases) [![Wheel](https://img.shields.io/pypi/wheel/paperank)](https://pypi.org/project/paperank/)

Quality and tooling:

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=python)](https://github.com/psf/black) [![Ruff](https://img.shields.io/badge/lint-ruff-1f79ff?logo=python)](https://github.com/astral-sh/ruff) [![mypy](https://img.shields.io/badge/type--checked-mypy-blue?logo=python)](https://github.com/python/mypy)

<!-- [![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/gwr3n/paperank?label=openssf%20scorecard)](https://securityscorecards.dev/viewer/?uri=github.com/gwr3n/paperank) -->

<!-- [![pre-commit.ci](https://results.pre-commit.ci/badge/github/gwr3n/paperank/main.svg)](https://results.pre-commit.ci/latest/github/gwr3n/paperank/main)  -->

Project/community:

[![Issues](https://img.shields.io/github/issues/gwr3n/paperank)](https://github.com/gwr3n/paperank/issues) [![PRs](https://img.shields.io/github/issues-pr/gwr3n/paperank)](https://github.com/gwr3n/paperank/pulls) [![Stars](https://img.shields.io/github/stars/gwr3n/paperank?style=social)](https://github.com/gwr3n/paperank/stargazers)

Docs:

[![Docs](https://img.shields.io/badge/docs-site-blue)](https://github.com/gwr3n/paperank)

**A Publication Ranking and Citation Network Analysis Tools**

`paperank` is a Python package for analyzing scholarly impact using citation networks. It provides tools to build citation graphs from DOIs, compute PapeRank (a PageRank-like score), fetch publication metadata, and export ranked results. The package is designed for researchers, bibliometricians, and developers interested in quantifying publication influence within local or global citation networks ([use cases](docs/Use%20cases.md)).

For a discussion on the use of PageRank-like scores beyond the web see [[Gleich, 2014](https://arxiv.org/abs/1407.5107)].

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
from paperank import crawl_and_rank_frontier

# Set your target DOI
doi = "10.1016/j.ejor.2005.01.053"

# Run the analysis
results = crawl_and_rank_frontier(
    doi=doi,
    steps=2,
    output_format="json"  # or "csv"
)
```

This will:
- Collect the citation neighborhood around the DOI with 2 iterative crawl steps (each step uses 1-hop neighborhoods)
- Compute PapeRank scores
- Save results to a file (`<DOI>.json` or `<DOI>.csv`)

---

## Advanced Parameters

You can fine-tune the crawl and ranking via the following parameters:

- `min_year`: Optional minimum publication year to include during crawling (filters older works).
- `min_citations`: Optional minimum total citation count to include during crawling (filters low-signal works).
- `alpha`: PageRank damping factor (default `0.85`).
- `tol`: Convergence tolerance for the power iteration (default `1e-12`).
- `max_iter`: Maximum number of power-iteration steps (default `10000`).
- `teleport`: Optional teleportation distribution (NumPy array of size N), non-negative and summing to 1. If `None`, a uniform distribution is used.

Example:

```python
from paperank import crawl_and_rank_frontier

results = crawl_and_rank_frontier(
    doi="10.1016/j.ejor.2005.01.053",
    steps=1,
    min_year=2000,       
    min_citations=5,     
    alpha=0.85,
    tol=1e-12,
    max_iter=20000,
    teleport=None
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