"""Which embedder config selects, and the contract both of them keep."""

from __future__ import annotations

import pytest

from rag.config.settings import IndexSettings
from rag.index.embed import (
    BGE_M3,
    BGE_SMALL,
    BGEM3Embedder,
    SmallHybridEmbedder,
    build_embedder,
)
from rag.index.lexical import lexical_sparse, token_id


def test_the_default_is_the_small_model():
    """Speed is the default because an hour and a half per document is not a
    system anyone can run. The better model is one config line away."""
    assert isinstance(build_embedder(IndexSettings()), SmallHybridEmbedder)


def test_bge_m3_is_still_selectable():
    settings = IndexSettings(embed_model=BGE_M3, embed_dims=1024)
    assert isinstance(build_embedder(settings), BGEM3Embedder)


def test_the_embedder_reports_the_width_config_declared():
    """`embed_dims` creates the Qdrant collection. An embedder that disagreed
    with it would write vectors the collection rejects."""
    settings = IndexSettings(embed_model=BGE_SMALL, embed_dims=384)
    assert build_embedder(settings).dims == 384


def test_the_model_name_is_the_one_that_was_asked_for():
    """It is stamped on every chunk, and that stamp is what a re-embed
    backfill selects on."""
    settings = IndexSettings(embed_model=BGE_SMALL, embed_dims=384)
    assert build_embedder(settings).model_name == BGE_SMALL


@pytest.mark.integration
async def test_the_small_embedder_returns_both_halves():
    """Hybrid retrieval needs a sparse vector, and this dense model does not
    produce one. Downloads the model, so it is an integration test."""
    settings = IndexSettings(embed_model=BGE_SMALL, embed_dims=384)
    embedded = await build_embedder(settings).embed(["turbine flange tolerance"])
    assert len(embedded[0].dense) == 384
    assert embedded[0].sparse


@pytest.mark.integration
async def test_the_query_prefix_never_reaches_the_sparse_side():
    """`bge-small` wants an instruction prefix on queries. Those words are an
    instruction to the dense model, and as lexical terms they would match
    every document in the corpus."""
    embedder = SmallHybridEmbedder(IndexSettings(embed_model=BGE_SMALL, embed_dims=384))
    embedded = await embedder.embed_queries(["turbine flange"])
    assert embedded[0].sparse == lexical_sparse("turbine flange")
    assert token_id("represent") not in embedded[0].sparse
