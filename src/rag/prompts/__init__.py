"""Versioned prompt files plus a registry."""

from rag.prompts.registry import (
    Prompt,
    PromptNotFoundError,
    PromptRegistry,
    get_registry,
)
from rag.prompts.render import (
    RenderedContext,
    StrippedMarker,
    assemble,
    new_nonce,
    render_context,
    strip_delimiters,
)
from rag.prompts.validate import (
    AnswerPayload,
    Citation,
    ValidationOutcome,
    ValidationReport,
    fallback_answer,
    validate,
)

__all__ = [
    "AnswerPayload",
    "Citation",
    "Prompt",
    "PromptNotFoundError",
    "PromptRegistry",
    "RenderedContext",
    "StrippedMarker",
    "ValidationOutcome",
    "ValidationReport",
    "assemble",
    "fallback_answer",
    "get_registry",
    "new_nonce",
    "render_context",
    "strip_delimiters",
    "validate",
]
