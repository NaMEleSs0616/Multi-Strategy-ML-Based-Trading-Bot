"""
Alpaca live WebSocket ingest (placeholder).

Set ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` in the environment, then implement
subscription to ``IEX``/``SIP`` bars and call ``BarStore.upsert_bars`` on each message.
"""

from __future__ import annotations

import os
from typing import Optional


def is_alpaca_configured() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def run_live_stream(
    symbols: list[str],
    *,
    db_path: Optional[str] = None,
) -> None:
    if not is_alpaca_configured():
        raise RuntimeError("Alpaca API keys not set in environment")
    raise NotImplementedError("Alpaca WebSocket ingest is not wired yet — use sync_data.py for now.")
