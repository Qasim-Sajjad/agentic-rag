"""URL canonicalization. The first of three exact dedup points.

Canonicalizing before the frontier is what stops the same page being fetched
under five tracking parameters.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PREFIXES = ("utm_",)
TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src"}
)
DEFAULT_PORTS = {"http": "80", "https": "443"}


def is_tracking(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PREFIXES)


def canonicalize(url: str) -> str:
    """Lowercase host, drop the fragment and tracking params, sort the rest."""
    parts = urlsplit(url.strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not is_tracking(k)
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            _netloc(parts.scheme.lower(), parts.netloc.lower()),
            parts.path or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def _netloc(scheme: str, netloc: str) -> str:
    host, _, port = netloc.partition(":")
    if port and DEFAULT_PORTS.get(scheme) == port:
        return host
    return netloc


def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize(url).encode("utf-8")).hexdigest()


def raw_content_hash(content: bytes) -> str:
    """Second dedup point: identical bytes never reach extraction twice."""
    return hashlib.sha256(content).hexdigest()
