"""VLM OCR. Interface and routing gate implemented, inference stubbed.

Two rules are written down here because they are the design, and whoever wires
up a GPU should not have to rediscover them:

1. Pass any extracted text layer alongside the image, even a poor one. The
   model aligns to it instead of free generating.
2. Set confidence below 1.0 on every OCR block. VLM errors are fluent and pass
   spell checks, so they reach the index looking correct. Confidence
   propagation is the mitigation, not accuracy.
"""

from __future__ import annotations

from rag.extract.protocols import ParserUnavailableError
from rag.extract.types import CanonicalDoc

OCR_CONFIDENCE = 0.7


class VLMOCRParser:
    name = "vlm_ocr"
    version = "0.0-stub"

    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc:
        raise ParserUnavailableError(
            "VLM OCR is a stub. The routing gate that selects it is implemented "
            "and tested, no GPU inference is stood up. Wire an endpoint in "
            "config and implement VLMOCRParser.parse to enable it"
        )
