"""Demonstration client. Connects, lists both tools with schemas, calls both.

    python -m rag.mcp.client

A deliverable rather than a test helper: it proves discovery is live, not
hardcoded, which is the only reason the MCP boundary is worth its overhead.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rag.config.settings import get_settings
from rag.log import configure_logging, get_logger

log = get_logger("mcp.client")

DEMO_QUERY = "what risks were disclosed"


def server_url() -> str:
    settings = get_settings()
    return f"http://127.0.0.1:{settings.mcp.port}/mcp"


async def describe_tools(session: Any) -> list[str]:
    listed = await session.list_tools()
    for tool in listed.tools:
        log.info(
            "tool discovered",
            name=tool.name,
            description=(tool.description or "").strip().splitlines()[0],
            schema=json.dumps(tool.input_schema)[:400],
        )
    return [tool.name for tool in listed.tools]


async def call_both(session: Any) -> None:
    search = await session.call_tool(
        "search_corpus", {"request": {"query": DEMO_QUERY, "top_k": 3}}
    )
    log.info("search_corpus result", output=_preview(search))
    status = await session.call_tool("get_ingest_status", {"request": {}})
    log.info("get_ingest_status result", output=_preview(status))


def _preview(result: Any) -> str:
    if getattr(result, "structuredContent", None):
        return json.dumps(result.structuredContent)[:600]
    blocks = getattr(result, "content", [])
    return "".join(getattr(block, "text", "") for block in blocks)[:600]


async def run() -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(server_url()) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        names = await describe_tools(session)
        await call_both(session)
        return names


def main() -> None:
    configure_logging()
    names = asyncio.run(run())
    log.info("discovery complete", tools=names)


if __name__ == "__main__":
    main()
