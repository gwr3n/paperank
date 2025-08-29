import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .doi_utils import normalize_doi, doi_to_path_segment
from typing import Dict, List, Any
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
        "User-Agent": "paperank/0.1 (+https://github.com/gwr3n/paperank)",
        "Accept": "application/json",
    })
    return s

# Thread-local session to safely reuse connections in concurrent calls
_TLS = local()

def _get_session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = _session()
        _TLS.session = s
    return s

def get_citing_dois(doi: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Query the OpenCitations COCI API for articles citing the given DOI.

    Args:
        doi: The DOI of the article to query.
        timeout: Timeout for the HTTP request in seconds.

    Returns:
        dict: A dictionary with keys:
            - "article_doi": The normalized input DOI.
            - "citing_dois": List of DOIs citing the input DOI.

    Example output:
        {
            "article_doi": "10.1016/j.ejor.2016.12.001",
            "citing_dois": [
                "10.1080/1540496x.2019.1696189",
                "10.1016/j.intfin.2017.09.008",
                ...
            ]
        }
    The list may be empty if there are no citing articles found for the given DOI.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    doi_seg: str = doi_to_path_segment(doi)
    url: str = f"https://opencitations.net/index/coci/api/v1/citations/{doi_seg}"
    response = _get_session().get(url, timeout=timeout)
    response.raise_for_status()
    data: List[dict] = response.json() or []
    seen: set = set()
    citing_dois: List[str] = []
    for item in data:
        d = item.get("citing")
        if not d:
            continue
        d = normalize_doi(d)
        if d not in seen:
            seen.add(d)
            citing_dois.append(d)
    return {"article_doi": normalize_doi(doi), "citing_dois": citing_dois}