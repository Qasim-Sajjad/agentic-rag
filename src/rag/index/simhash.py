"""Near duplicate detection. 64 bit SimHash over word 5-grams, banded index.

Runs on extracted text only, never on raw HTML. Every page on a site shares
nav, footer and sidebar, which is most of the shingle space of the raw markup,
so raw HTML SimHash marks every page on a domain as a duplicate of every other.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

BITS = 64
SHINGLE_SIZE = 5
BAND_COUNT = 4
BAND_BITS = BITS // BAND_COUNT
BAND_MASK = (1 << BAND_BITS) - 1


def shingles(text: str, size: int = SHINGLE_SIZE) -> list[str]:
    words = text.split()
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def simhash(text: str) -> int:
    vector = [0] * BITS
    for shingle in shingles(text):
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(BITS):
            vector[bit] += 1 if value >> bit & 1 else -1
    return sum(1 << bit for bit in range(BITS) if vector[bit] > 0)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def bands(value: int) -> list[tuple[int, int]]:
    """Band index and its bits. Candidates must share at least one band."""
    return [(i, (value >> (i * BAND_BITS)) & BAND_MASK) for i in range(BAND_COUNT)]


class SimHashIndex:
    """Banded index, so a lookup is not a scan of every document seen."""

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._bands: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._hashes: dict[str, int] = {}

    def add(self, key: str, value: int) -> None:
        self._hashes[key] = value
        for band in bands(value):
            self._bands[band].add(key)

    def candidates(self, value: int) -> set[str]:
        found: set[str] = set()
        for band in bands(value):
            found |= self._bands.get(band, set())
        return found

    def find_duplicate(self, value: int) -> str | None:
        """First key within the Hamming threshold, or None."""
        for key in self.candidates(value):
            if hamming(self._hashes[key], value) <= self._threshold:
                return key
        return None

    def __len__(self) -> int:
        return len(self._hashes)
