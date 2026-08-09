"""Shared test harness. The fixture server runs on a real socket, not an ASGI
transport, because the fetch tiers it exists to test drive TLS and a browser.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from tests.fixtures.server import create_app

STARTUP_TIMEOUT_SECONDS = 10.0
POLL_SECONDS = 0.01

# Wait for the FastApi test Server to Start.
def _wait_until_started(server: uvicorn.Server) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("fixture server did not start")
        time.sleep(POLL_SECONDS)

# Pytest fixture to start the FastApi test server and yield the base URL.
@pytest.fixture(scope="session")
def fixture_server() -> Iterator[str]:
    """Base URL of a running fixture server on an ephemeral port."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        _wait_until_started(server)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=STARTUP_TIMEOUT_SECONDS)


@pytest.fixture
def client(fixture_server: str) -> Iterator[httpx.Client]:
    """Client with server state reset, so attempt counters do not leak."""
    with httpx.Client(base_url=fixture_server, timeout=5.0) as http:
        http.post("/__reset")
        yield http
