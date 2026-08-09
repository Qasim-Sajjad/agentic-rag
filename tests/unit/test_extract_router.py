"""Content routing: correct headers, wrong extensions, magic bytes only."""

from __future__ import annotations

import pytest

from rag.extract import router
from rag.extract.protocols import UnsupportedTypeError
from rag.extract.service import ExtractService

PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer"
HTML_BYTES = b"<!doctype html><html><body><p>hello</p></body></html>"


def test_header_is_used_when_it_agrees():
    assert router.resolve_mime("text/html; charset=utf-8", HTML_BYTES) == router.HTML


def test_charset_is_stripped_from_the_header():
    assert router.normalize_mime("application/pdf; qs=0.9") == router.PDF


def test_magic_bytes_win_over_a_wrong_header():
    """A URL like /download?id=8821 says html and returns a PDF."""
    assert router.resolve_mime("text/html", PDF_BYTES) == router.PDF


def test_octet_stream_falls_back_to_magic_bytes():
    assert router.resolve_mime("application/octet-stream", PDF_BYTES) == router.PDF


def test_octet_stream_on_html_falls_back_to_magic_bytes():
    assert router.resolve_mime("application/octet-stream", HTML_BYTES) == router.HTML


def test_a_missing_header_falls_back_to_magic_bytes():
    assert router.resolve_mime("", PDF_BYTES) == router.PDF


def test_an_unknown_type_is_reported_not_raised_blindly():
    service = ExtractService()
    with pytest.raises(UnsupportedTypeError) as caught:
        service.parser_for("application/x-nonsense", b"\x00\x01\x02")
    assert caught.value.mime == "application/x-nonsense"


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        (router.HTML, "trafilatura"),
        (router.PDF, "pdf_router"),
        (router.CSV, "tabular"),
        (router.PLAIN, "plaintext"),
    ],
)
def test_registry_maps_each_type_to_its_parser(mime, expected):
    assert parser_name(mime) == expected


def parser_name(mime: str) -> str:
    from rag.extract.service import parser_registry

    return parser_registry()[mime].name
