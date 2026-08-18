"""Lexical sparse vectors, computed in Python rather than by a model.

BGE-M3 emits dense and sparse from one forward pass, which is why hybrid
retrieval was one model. A small dense only model is roughly fifty times faster
on CPU, but taking it would drop the sparse side of the index, and the lexical
half is what finds a course code, a part number or a surname that no embedding
places near the query.

So the sparse side moves here. This is a hashed bag of words with sublinear term
frequency, the scoring half of BM25 without the document length normalisation:
Qdrant scores the dot product of the two sparse vectors, and the query side is
one word per dimension, so what survives is the term weighting. It costs
microseconds per chunk and needs no model.

The hash is `blake2b`, not the builtin `hash`, which is salted per process and
would give a document one id at index time and a different one at query time.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

#: Qdrant sparse indices are unsigned 32 bit. Collisions at this width are rare
#: enough to be noise next to the dense side, and a wider space buys nothing.
VOCAB = 2**20

_WORD = re.compile(r"[a-z0-9][a-z0-9._-]*")

#: The words that appear in almost every chunk carry no signal and would
#: dominate the dot product. Deliberately short: a stopword list that removes a
#: real query term is worse than one that keeps a common one.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "there",
        "their",
        "they",
        "you",
        "your",
        "we",
        "our",
        "i",
    ]
)


def token_id(word: str) -> int:
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % VOCAB


def lexical_sparse(text: str) -> dict[int, float]:
    """Sublinear term frequency, L2 normalised.

    `1 + log(tf)` rather than raw counts, because a word repeated forty times in
    a table is not forty times more relevant than one mentioned once, and raw
    counts let one such chunk outrank everything for its own boilerplate.
    """
    counts = Counter(
        word
        for word in _WORD.findall(text.lower())
        if word not in STOPWORDS and len(word) > 1
    )
    if not counts:
        return {}
    weights = {token_id(word): 1.0 + math.log(count) for word, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
    return {key: value / norm for key, value in weights.items()}
