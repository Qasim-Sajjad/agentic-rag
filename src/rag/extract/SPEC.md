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
    text: str  # tables are markdown
    level: int | None = None  # heading depth
    provenance: Provenance
    confidence: float = 1.0  # below 1.0 only from OCR


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

Unknown type writes a row to the `dead_letter` table defined in
`src/rag/fetch/SPEC.md` with `stage = 'extract'`, `reason =
'unsupported_type'`, and the observed MIME in `detail`. Extending support is
one registry entry.

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
| Text layer, tables or multi column | `PyMuPDF4LLMParser` |
| No usable text layer | `VLMOCRParser` |

Both text classes route to `PyMuPDF4LLMParser`. `pymupdf4llm` already emits
tables as Markdown, and measured against it Docling cost roughly ten times as
much per page for output the chunker treats identically. Gate 2 is still
computed, because `page_class` is what the trace reports and what an OCR
decision is made from, but it no longer selects a different parser.

`pymupdf4llm` must be configured with `use_layout(False)`, which
`configure_layout` in `src/rag/extract/pdf.py` sets from
`extract.pymupdf_use_layout` (default false). With `pymupdf-layout` and
`rapidocr` installed, `pymupdf4llm` otherwise takes a GNN layout and OCR path on
every page whether or not it has a text layer, silently overriding gate 1.
Measured on a real report: 2583 ms per page with, 227 ms without, for 0.7 percent
less text. That switch is the difference between a 500 page document taking
forty minutes and taking four.

## Page range parallelism

Split every PDF into ranges of `pages_per_task` (default 50) before parsing.
Each range is an independent task with its own checkpoint. A 1000 page document
becomes 20 tasks, not one long worker lock.

A range breaks on a change of parser, not on a change of page class. Both text
classes run `PyMuPDF4LLMParser`, so only the scanned boundary is real. Splitting
on the table boundary as well turned a 252 page prospectus that alternates prose
and tables into 64 ranges: 64 parser instances, each reopening the document, for
a distinction that selects nothing. The same document now plans 6.

Ranges run concurrently, bounded by `extract.max_parallel_ranges` (default 4).
Each range is synchronous PyMuPDF work and runs in a worker thread, as does the
initial probe: a 500 page extract on the event loop stalls every other request
the API is serving, which reads as a hung server rather than a slow one.

Every block carries the page it came from, counting from one, as a reader sees
it in a PDF viewer and as the OCR path already recorded it. `pymupdf4llm` is
called with `page_chunks=True`, which returns a Markdown entry per page with its
number attached at the same cost as one blob for the whole range. Attributing a
whole range to its first page was tolerable when a range was a few pages and
became a lie once ranges merged to fifty.

Reassembly concatenates ranges in page order and runs one fixup pass: a table
at the top of a range with the same column count as the table ending the
previous range, and no header row, is merged into it. `gather` preserves input
order, so the fixup still sees a split table's two halves adjacent no matter
which range finished first.

## Progress

`PdfRouter.parse_progress(content, source_url, progress)` does the same work as
`parse` and reports each stage to the `Progress` sink in `src/rag/progress.py`:
`probe` once with the range count, then `extract` per completed range. `parse`
itself stays two arguments, so the `DocumentParser` protocol every other parser
implements is unchanged, and `ExtractService` looks for `parse_progress` the way
retrieval looks for `embed_queries`. A minutes long extract that reports nothing
cannot be told apart from one that has hung.

## Page furniture

A faxed clinical note carries the patient banner at the top of every page, the
sending machine's line at the foot of every page, and a form artifact wherever a
field was left blank. A prospectus carries a running header. All of it is
printed on the page, so the text layer and OCR both read it, and none of it is
content.

`strip_repeated` in `src/rag/extract/boilerplate.py` removes it after
reassembly, where the whole document is visible: a single range cannot tell a
banner from a sentence. A block is furniture when it is a paragraph of at most
`repeat_max_chars` (default 200) whose text, with digits masked, appears at
least `repeat_min_count` (default 3) times. Digits are masked because the page
number, the timestamp and the fax counter change per page while the line does
not.

Headings and tables are never dropped. A repeated heading is document
structure, and the sections under it would lose their section path; a repeated
table is data the document contains.

Measured on a five page faxed consultation: 118 blocks to 77, and the first
chunk went from pure banner to the medication list. Furniture also collides on
`chunk_hash`, so banner only chunks were being dropped by dedup and the indexed
count stopped describing the document.

## OCR

`VLMOCRParser` sends the page range to Claude as a `document` block and takes
back Markdown. The API renders the pages itself, so there is no image pipeline
here and no resolution constant to get wrong. Model and budgets come from
`extract.ocr`; the key is the one already in `.env` for the agent.

Citations are enabled on the document block because the per page numbers they
return are what fills `Provenance.page`. Without them one blob of transcribed
Markdown has no page attribution at all. A block whose text carries no citation
keeps the range's first page rather than inheriting a neighbour's, because a
wrong page number is worse than a coarse one.

Transcription is routed through the same Markdown block parser the text layer
path uses, so an OCR'd table is the same `Block` shape as a parsed one and
nothing downstream can tell which produced it. `confidence` is the one field
that can.

Two rules:

- Pass any extracted text layer alongside the image, even a poor one. The model
  aligns to it rather than free generating. This is the main defence against
  plausible-looking wrong output.
- Set `confidence` below 1.0 on every OCR block so downstream stages can hedge.

VLM output is not a normal OCR error mode. Errors are fluent and pass spell
checks, so they reach the index looking correct. Confidence propagation is the
mitigation, not accuracy alone.

A third rule emerged from running it. **A page with no text produces no blocks.**
Prompt v1 answered a page that was a photograph with `[Image: a laptop keyboard
resting on a marble surface]`, which is a model authored sentence entering the
corpus as document content and citable as if the document had said it.
Describing a picture is not transcription. v2 forbids it, and the rule is pinned
by a test rather than left to the prompt.

A range that cannot be read is skipped, not fatal, which is the same rule the
rest of the ladder follows: OCR disabled, no API key, or a safety refusal all
return no blocks. A scanned appendix must not cost a document its readable
pages.

## Tests

Unit:
- Content routing picks the right parser for correct headers, wrong extensions,
  and magic bytes only
- Gate 1 classifies a digital born PDF, a scanned PDF, and one with broken font
  encoding
- Reassembly merges a table split across a page range boundary
- A table page does not start a new range, a scanned page does, and a merged
  text run still respects `pages_per_task`
- Every page's blocks carry that page's number, counting from one
- A banner repeated on every page is dropped, with digits masked so a per page
  footer still matches, while a repeated heading, a repeated table and a long
  repeated paragraph are all kept
- The probe reports each page as it reads it, and extraction announces itself
  before the first range finishes
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

- `Block.confidence` is a constant 0.7 on OCR output, not a calibrated score. A
  VLM does not return one, and a self reported number would be worse than an
  honest constant. It marks provenance, not reliability.
- OCR is billed per page and has no spend cap. A large scanned corpus would be
  expensive, and nothing here stops it.
- OCR output is not evaluated. There is no gold transcription set, so accuracy
  on tables and multi column layout is unmeasured.
- `DoclingParser` is still wired for DOCX and XLSX but no longer reachable from
  the PDF ladder, so the COMPLEX_TEXT page class no longer changes what runs.
  Gate 2's table detection costs roughly 240 ms per page for a distinction that
  now only appears in the trace.
- Div based HTML table layouts are not reconstructed.
- Images and figures are captured as captions only. No image embedding.
