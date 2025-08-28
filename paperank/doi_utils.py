import re
from urllib.parse import quote, unquote

_DOI_PREFIX_RE = re.compile(r'^https?://(dx\.)?doi\.org/', re.IGNORECASE)

def normalize_doi(doi: str) -> str:
    """
    Normalize a DOI string by:
      - Removing any leading 'http(s)://doi.org/' or 'http(s)://dx.doi.org/' prefix.
      - Decoding percent-encoded characters.
      - Stripping whitespace.
      - Converting to lowercase.

    Args:
        doi: The DOI string to normalize.

    Returns:
        str: The normalized DOI string. If input is empty or None, returns as is.

    Example:
        >>> normalize_doi("https://doi.org/10.1016/j.ejor.2016.12.001")
        '10.1016/j.ejor.2016.12.001'
    """
    if not doi:
        return doi
    d: str = unquote(doi.strip())
    d = _DOI_PREFIX_RE.sub('', d)
    return d.lower()

def doi_to_path_segment(doi: str) -> str:
    """
    Convert a DOI string to a URL-safe path segment suitable for use in REST API endpoints.

    Args:
        doi: The DOI string to convert.

    Returns:
        str: The DOI as a URL-encoded path segment.

    Example:
        >>> doi_to_path_segment("10.1016/j.ejor.2016.12.001")
        '10.1016%2Fj.ejor.2016.12.001'
    """
    return quote(normalize_doi(doi), safe='')