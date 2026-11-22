"""Lightweight zero-dependency local Web UI & REST/SSE server."""

from .server import create_server, run_server

__all__ = ["create_server", "run_server"]
