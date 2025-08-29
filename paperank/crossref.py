import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .doi_utils import normalize_doi, doi_to_path_segment
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from threading import local

def _session() -> requests.Session:
    """
    Create and configure a requests.Session with retry logic for robust HTTP requests.

    Returns:
        requests.Session: Configured session object.
    """
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": f"paperank/0.1 (+https://github.com/gwr3n/paperank)",
        "Accept": "application/json",
    })
    return s

# Thread-local session for safe reuse in concurrency
_TLS = local()

def _get_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = _session()
        _TLS.session = s
    return s

@lru_cache(maxsize=200_000)
def get_work_metadata(doi: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Retrieve metadata for a work from the Crossref API given a DOI.

    Args:
        doi: The DOI of the work to retrieve metadata for.
        timeout: Timeout for the HTTP request in seconds.

    Returns:
        dict: A shallow copy of the Crossref metadata envelope for the work.
        Both the top-level envelope and its 'message' dict (if present) are
        shallow-copied to avoid accidental mutation of cached objects by callers.

    The returned JSON contains:
      - status: API response status.
      - message-type: Type of message (usually 'work').
      - message-version: Version of the message format.
      - message: Main metadata object, including:
        - indexed: Indexing info (date, timestamp, version).
        - reference-count: Number of references cited.
        - publisher: Publisher name.
        - issue, volume, page: Journal issue, volume, and page range.
        - license: List of license objects (start date, URL, etc.).
        - content-domain: Domains and crossmark restriction.
        - short-container-title, container-title: Journal titles.
        - published-print, published: Publication dates.
        - DOI: Article DOI.
        - type: Work type (e.g., 'journal-article').
        - created: Record creation date.
        - update-policy: Update policy URL.
        - source: Metadata source.
        - is-referenced-by-count: Number of times cited.
        - title: Article title.
        - prefix: DOI prefix.
        - author: List of authors (names, ORCID, affiliations).
        - member: Crossref member ID.
        - reference: List of references cited (with metadata).
        - original-title, subtitle, short-title: Titles/subtitles.
        - language: Article language.
        - link: List of content links.
        - deposited: Metadata deposit date.
        - score: Relevance score.
        - resource: Primary resource URL.
        - issued: Publication date.
        - references-count: Number of references.
        - journal-issue: Journal issue details.
        - alternative-id: Alternative identifiers.
        - URL: DOI URL.
        - relation: Related works.
        - ISSN, issn-type: Journal ISSNs.
        - subject: Subject areas.
        - assertion: List of assertions (publisher, copyright, etc.).

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    doi_seg: str = doi_to_path_segment(doi)
    url: str = f"https://api.crossref.org/works/{doi_seg}"
    response = _get_session().get(url, timeout=timeout)
    response.raise_for_status()
    data: Any = response.json()
    if isinstance(data, dict):
        out = data.copy()
        msg = out.get("message")
        if isinstance(msg, dict):
            out["message"] = msg.copy()
        return out
    return data

# Cache management
def clear_caches() -> None:
    """Clear LRU cache for Crossref metadata requests."""
    try:
        get_work_metadata.cache_clear()
    except Exception:
        pass

def get_cited_dois(doi: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Given an article DOI, returns a dictionary with the normalized DOI and a list of DOIs it cites.

    Args:
        doi: The DOI of the article to query.
        timeout: Timeout for the HTTP request in seconds.

    Returns:
        dict: {
            "article_doi": <normalized DOI>,
            "cited_dois": [list of normalized DOIs]
        }

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    metadata: Dict[str, Any] = get_work_metadata(doi, timeout=timeout)
    references: List[Dict[str, Any]] = metadata.get("message", {}).get("reference", []) or []
    seen: set = set()
    cited_dois: List[str] = []
    for ref in references:
        d = ref.get("DOI")
        if not d:
            continue
        d = normalize_doi(d)
        if d not in seen:
            seen.add(d)
            cited_dois.append(d)
    return {"article_doi": normalize_doi(doi), "cited_dois": cited_dois}

def extract_authors_title_year(meta: Dict[str, Any]) -> Tuple[List[str], str, Optional[int]]:
    """
    Extract authors (list of strings), title (string), and year (int or None) from
    a Crossref metadata envelope or message dict.

    Args:
        meta: Metadata dictionary from Crossref (envelope or message).

    Returns:
        tuple: (authors, title, year)
            authors: List of author names (strings).
            title: Title string.
            year: Year as int, or None if not found.
    """
    def _first_title(m: Dict[str, Any]) -> str:
        title = m.get("title")
        if isinstance(title, list) and title:
            t = (title[0] or "").strip()
            if t:
                return t
        if isinstance(title, str) and title.strip():
            return title.strip()
        st = m.get("short-title")
        if isinstance(st, list) and st:
            t = (st[0] or "").strip()
            if t:
                return t
        sub = m.get("subtitle")
        if isinstance(sub, list) and sub:
            t = (sub[0] or "").strip()
            if t:
                return t
        return "Unknown title"

    def _extract_year_from_dateobj(dobj: Any) -> Optional[int]:
        if not isinstance(dobj, dict):
            return None
        dp = dobj.get("date-parts") or dobj.get("date_parts")
        if isinstance(dp, list) and dp and isinstance(dp[0], (list, tuple)) and dp[0]:
            y = dp[0][0]
            try:
                return int(y)
            except Exception:
                return None
        return None

    def _year(m: Dict[str, Any]) -> Optional[int]:
        for key in ("issued", "published-print", "published-online", "published", "created", "deposited"):
            y = _extract_year_from_dateobj(m.get(key))
            if y is not None:
                return y
        return None

    # Accept both envelope and raw message dict
    msg: Dict[str, Any] = {}
    if isinstance(meta, dict):
        msg = meta.get("message") if isinstance(meta.get("message"), dict) else meta
    m: Dict[str, Any] = msg if isinstance(msg, dict) else {}

    # Authors
    authors_list: List[str] = []
    authors = m.get("author") or []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict):
                name = (a.get("name") or a.get("literal") or "").strip()
                if not name:
                    given = (a.get("given") or "").strip()
                    family = (a.get("family") or "").strip()
                    name = f"{given} {family}".strip()
                if name:
                    authors_list.append(name)
            elif isinstance(a, str) and a.strip():
                authors_list.append(a.strip())

    title_str: str = _first_title(m)
    year_val: Optional[int] = _year(m)
    return authors_list, title_str, year_val

if __name__ == "__main__":
    test_doi = "10.1016/j.ejor.2016.12.001"
    test_type = "cited"  

    if test_type == "metadata":
        result = get_work_metadata(test_doi)
        print(f"Metadata for {test_doi}:")
        print(result)
    elif test_type == "cited":
        cited = get_cited_dois(test_doi)
        print(f"Cited DOIs by {test_doi}:")
        print(cited)
    else:
        print("Unknown test type. Use 'metadata' or 'cited'.")