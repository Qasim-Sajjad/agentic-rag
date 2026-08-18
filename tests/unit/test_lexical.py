"""The sparse half of hybrid retrieval when the dense model does not emit one."""

from __future__ import annotations

import math

from rag.index.lexical import STOPWORDS, lexical_sparse, token_id


def test_the_same_word_always_hashes_to_the_same_id():
    """The builtin `hash` is salted per process, which would give a document one
    id at index time and a different one at query time."""
    assert token_id("turbine") == token_id("turbine")


def test_different_words_get_different_ids():
    assert token_id("turbine") != token_id("actuator")


def test_case_is_not_a_distinction():
    assert lexical_sparse("Turbine") == lexical_sparse("turbine")


def test_stopwords_are_dropped():
    """They appear in nearly every chunk, so they would dominate the dot
    product without separating anything."""
    assert token_id("the") not in lexical_sparse("the turbine and the flange")
    assert "the" in STOPWORDS


def test_a_repeated_word_outweighs_a_single_mention():
    sparse = lexical_sparse("turbine turbine turbine flange")
    assert sparse[token_id("turbine")] > sparse[token_id("flange")]


def test_repetition_is_sublinear():
    """Forty mentions is not forty times the relevance. Raw counts let one
    boilerplate heavy chunk outrank everything for its own repetition."""
    many = lexical_sparse(" ".join(["turbine"] * 40) + " flange")
    ratio = many[token_id("turbine")] / many[token_id("flange")]
    assert ratio < 10


def test_the_vector_is_unit_length():
    """Normalised, so a long chunk does not outscore a short one on length."""
    sparse = lexical_sparse("turbine flange bearing housing tolerance")
    assert math.isclose(math.sqrt(sum(v * v for v in sparse.values())), 1.0)


def test_a_long_chunk_does_not_dominate_a_short_one():
    short = lexical_sparse("turbine flange")
    long = lexical_sparse("turbine flange " + " ".join(f"word{i}" for i in range(200)))
    assert short[token_id("turbine")] > long[token_id("turbine")]


def test_a_query_shares_ids_with_the_passage_that_answers_it():
    """The whole point: index side and query side must agree on the id space."""
    passage = lexical_sparse("The impeller shroud tolerance is 0.18 millimetres.")
    query = lexical_sparse("impeller shroud tolerance")
    assert set(query) & set(passage) == set(query)


def test_text_with_nothing_but_stopwords_produces_an_empty_vector():
    """Empty, not a vector of noise. Qdrant scores it against nothing."""
    assert lexical_sparse("the and of to") == {}


def test_identifiers_survive_tokenisation():
    """A course code or a part number is exactly what the lexical side is for,
    and a tokeniser that splits on punctuation would destroy it."""
    sparse = lexical_sparse("Course CS-401.2 is offered in autumn")
    assert token_id("cs-401.2") in sparse
