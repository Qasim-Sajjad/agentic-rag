"""URL canonicalization and SimHash, including the raw HTML failure mode."""

from __future__ import annotations

from rag.index.simhash import SimHashIndex, hamming, simhash
from rag.index.urls import canonicalize, url_hash

BOILERPLATE = (
    "<nav>Home About Careers Contact Products Pricing Blog Login</nav>"
    "<footer>Copyright 2026. All rights reserved. Terms Privacy Cookies</footer>"
) * 4


def test_tracking_params_are_stripped():
    assert canonicalize("https://a.test/p?utm_source=x&id=7") == "https://a.test/p?id=7"


def test_fbclid_is_stripped():
    assert canonicalize("https://a.test/p?fbclid=abc") == "https://a.test/p"


def test_query_params_are_sorted():
    assert canonicalize("https://a.test/p?b=2&a=1") == "https://a.test/p?a=1&b=2"


def test_the_host_is_lowercased():
    assert canonicalize("https://A.TEST/p") == "https://a.test/p"


def test_the_fragment_is_dropped():
    assert canonicalize("https://a.test/p#section") == "https://a.test/p"


def test_a_default_port_is_dropped():
    assert canonicalize("https://a.test:443/p") == "https://a.test/p"


def test_equivalent_urls_hash_the_same():
    left = url_hash("https://a.test/p?utm_source=x&b=2&a=1")
    right = url_hash("https://a.test/p?a=1&b=2")
    assert left == right


def test_identical_text_has_distance_zero():
    text = "Revenue rose nine percent against the same quarter last year."
    assert hamming(simhash(text), simhash(text)) == 0


def test_a_near_duplicate_is_within_the_threshold():
    base = "Revenue rose nine percent against the same quarter last year. " * 5
    edited = base.replace("nine", "ten", 1)
    assert hamming(simhash(base), simhash(edited)) <= 3


def test_unrelated_text_is_outside_the_threshold():
    left = "Revenue rose nine percent against the same quarter last year. " * 5
    right = "The cat sat on the mat while the dog slept by the fire. " * 5
    assert hamming(simhash(left), simhash(right)) > 3


def test_boilerplate_dominates_raw_html_and_hides_real_differences():
    """Why near-dedup runs on extracted text only, never on raw HTML."""
    left = simhash(BOILERPLATE + "<p>Revenue rose nine percent this quarter.</p>")
    right = simhash(BOILERPLATE + "<p>The board approved a share buyback.</p>")
    assert hamming(left, right) <= 3


def test_extracted_text_separates_the_same_two_pages():
    left = simhash("Revenue rose nine percent this quarter. " * 5)
    right = simhash("The board approved a share buyback. " * 5)
    assert hamming(left, right) > 3


def test_the_index_finds_a_near_duplicate():
    index = SimHashIndex(threshold=3)
    base = "Revenue rose nine percent against the same quarter last year. " * 5
    index.add("d1", simhash(base))
    assert index.find_duplicate(simhash(base.replace("nine", "ten", 1))) == "d1"


def test_the_index_does_not_match_unrelated_text():
    index = SimHashIndex(threshold=3)
    index.add("d1", simhash("Revenue rose nine percent. " * 5))
    assert index.find_duplicate(simhash("The cat sat on the mat. " * 5)) is None
