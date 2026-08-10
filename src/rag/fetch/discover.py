"""Link discovery inside a source.

The fetch SPEC says discovery inside a source is automatic and adding a domain
is not. This is the automatic half: links are followed only within the source's
own domain, so a link on page 40,000 can never enrol a new site.
"""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urldefrag, urljoin, urlsplit

from rag.index.urls import canonicalize

_HREF = re.compile(r"""href\s*=\s*["']([^"'#\s>]+)["']""", re.IGNORECASE)

SKIP_SCHEMES = ("mailto:", "javascript:", "tel:", "data:")

# Assets no parser handles. Following them costs a fetch and buys a dead letter
# row, which is noise in the failure counts rather than a real coverage gap.
SKIP_SUFFIXES = (
    ".zip",
    ".exe",
    ".dmg",
    ".css",
    ".js",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp4",
    ".webp",
)


def extract_links(
    content: bytes, base_url: str, domain: str, limit: int = 200
) -> list[str]:
    """Same domain links, canonicalized and deduplicated, in page order."""
    html = content.decode("utf-8", errors="replace")
    found: list[str] = []
    seen: set[str] = set()
    for match in _HREF.finditer(html):
        url = _resolve(match.group(1), base_url, domain)
        if url is not None and url not in seen:
            seen.add(url)
            found.append(url)
        if len(found) >= limit:
            break
    return found


def _resolve(href: str, base_url: str, domain: str) -> str | None:
    if href.lower().startswith(SKIP_SCHEMES):
        return None
    absolute, _ = urldefrag(urljoin(base_url, href))
    parts = urlsplit(absolute)
    return canonicalize(absolute) if _wanted(parts, domain) else None


def _wanted(parts: SplitResult, domain: str) -> bool:
    if parts.scheme not in ("http", "https"):
        return False
    if parts.netloc.lower().split(":")[0] != domain.lower():
        return False
    return not parts.path.lower().endswith(SKIP_SUFFIXES)
