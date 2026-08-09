# extract

Turns fetched bytes into `CanonicalDoc`, a single structured representation that
every downstream stage consumes. Chunking never knows which parser produced a
document.

## Contracts

```python
class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    FIGURE_CAPTION = "figure_caption"

class Provenance(BaseModel):
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    css_path: str | None = None

class Block(BaseModel):
    type: BlockType
    text: str                      # tables are markdown
    level: int | None = None       # heading depth
    provenance: Provenance
    confidence: float = 1.0        # below 1.0 only from OCR

class CanonicalDoc(BaseModel):
    doc_id: str
    source_url: str
    title: str | None
    published_at: date | None
    language: str
    blocks: list[Block]
    content_hash: str
    extractor_name: str
    extractor_version: str
```

```python
class DocumentParser(Protocol):
    name: str
    version: str
    async def parse(self, content: bytes, source_url: str) -> CanonicalDoc: ...
```

## Content routing

Route on response `Content-Type` plus magic bytes. Never on the URL extension.
URLs like `/download?id=8821` return PDFs.

```python
PARSER_REGISTRY: dict[str, DocumentParser] = {
    "text/html": TrafilaturaParser(),
    "application/pdf": PdfRouter(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DoclingParser(),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DoclingParser(),
    "text/csv": TabularParser(),
    "text/plain": PlainTextParser(),
}
```

Unknown type returns `UNSUPPORTED_TYPE` to the dead letter store with the
observed MIME recorded. Extending support is one registry entry.

## HTML

`trafilatura` for main content extraction with metadata (title, date, author).
Boilerplate removal happens here, and it must happen before near-dedup, because
nav and footer markup dominates the shingle space of raw HTML.

Tables are extracted separately and converted to markdown. Div based grid
layouts are not reconstructed. Record that as a gap.

## PDF ladder

`PdfRouter` picks a parser per page range, not per document. Real documents mix
digital born pages with scanned appendices.

**Gate 1, text layer.** Sample pages with PyMuPDF. Compute characters per page
and the ratio of replacement or non printable characters.
- Under `min_chars_per_page` (default 100), treat as scanned.
- Garbage ratio over `max_garbage_ratio` (default 0.2), treat as scanned.

**Gate 2, layout.** Detect column count and table presence.

Routing:

| Condition | Parser |
|---|---|
| Text layer, single column, no tables | `PyMuPDF4LLMParser` |
| Text layer, tables or multi column | `DoclingParser` |
| No usable text layer | `VLMOCRParser` |

Disable Docling's table and OCR models when gate 2 found neither. Running
TableFormer on prose is the largest avoidable cost in this stage.

## Page range parallelism

Split every PDF into ranges of `pages_per_task` (default 50) before parsing.
Each range is an independent task with its own checkpoint. A 1000 page document
becomes 20 tasks, not one long worker lock.

Reassembly concatenates ranges in page order and runs one fixup pass: a table
at the top of a range with the same column count as the table ending the
previous range, and no header row, is merged into it.

## OCR

`VLMOCRParser` renders pages to images and batches them to a VLM endpoint.
Two rules:

- Pass any extracted text layer alongside the image, even a poor one. The model
  aligns to it rather than free generating. This is the main defence against
  plausible-looking wrong output.
- Set `confidence` below 1.0 on every OCR block so downstream stages can hedge.

VLM output is not a normal OCR error mode. Errors are fluent and pass spell
checks, so they reach the index looking correct. Confidence propagation is the
mitigation, not accuracy alone.

## Tests

Unit:
- Content routing picks the right parser for correct headers, wrong extensions,
  and magic bytes only
- Gate 1 classifies a digital born PDF, a scanned PDF, and one with broken font
  encoding
- Reassembly merges a table split across a page range boundary
- Unknown MIME returns `UNSUPPORTED_TYPE` and does not raise

Extraction anchors, `evals/anchors/`: 12 pages spanning digital born, scanned,
two column, table heavy, and broken encoding. Each has 3 to 5 assertions:

```json
{"page": "annual_report_p14",
 "assert_contains": ["Risk Factors", "cybersecurity incident"],
 "assert_table_count": 2,
 "assert_table_0_rows": 7,
 "assert_cell": {"table": 0, "row": 3, "col": 1, "value": "1,204"}}
```

Anchors, not full transcription. They catch the failures that actually occur:
dropped tables, merged columns, lost headings.

## Known gaps

- `VLMOCRParser` is an interface with a stub. The routing gate is implemented
  and tested. No GPU inference is stood up.
- Div based HTML table layouts are not reconstructed.
- Images and figures are captured as captions only. No image embedding.
