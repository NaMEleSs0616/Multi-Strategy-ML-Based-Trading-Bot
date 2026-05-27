"""
Daily FinBERT sentiment from SEC filings (lightweight, cached).

Score in [-1, 1] per calendar day; PiT-shifted before merge.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "storage" / "sentiment_cache"
DEFAULT_MODEL = "ProsusAI/finbert"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_sentiment.json"


def _load_cache(symbol: str) -> dict[str, float]:
    path = _cache_path(symbol)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(symbol: str, scores: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(symbol).write_text(json.dumps(scores, indent=2), encoding="utf-8")


def _score_texts(texts: Sequence[str]) -> float:
    """Mean FinBERT sentiment in [-1, 1] (positive - negative)."""
    if not texts:
        return 0.0
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        logger.info("transformers not installed — sentiment defaults to 0")
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for text in texts[:20]:
            snippet = (text or "")[:512]
            if not snippet.strip():
                continue
            inputs = tokenizer(snippet, return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0).numpy()
            # FinBERT: 0=positive, 1=negative, 2=neutral
            scores.append(float(probs[0] - probs[1]))
    return float(np.mean(scores)) if scores else 0.0


def _download_filing_snippets(symbol: str, max_filings: int = 3) -> list[str]:
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        logger.info("sec-edgar-downloader not installed — skipping EDGAR fetch")
        return []

    dl_dir = CACHE_DIR / "edgar" / symbol.upper()
    dl_dir.mkdir(parents=True, exist_ok=True)
    try:
        dl = Downloader("MultiStrategyRLBot", "research@example.com", str(dl_dir.parent))
        dl.get("10-K", symbol, limit=max_filings, download_details=False)
        dl.get("10-Q", symbol, limit=max_filings, download_details=False)
    except Exception as exc:
        logger.warning("EDGAR download failed for %s: %s", symbol, exc)
        return []

    texts: list[str] = []
    for path in dl_dir.rglob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            texts.append(content[:4000])
        except Exception:
            continue
        if len(texts) >= max_filings * 2:
            break
    return texts


def daily_sentiment_series(
    symbol: str,
    index: pd.DatetimeIndex,
    *,
    pit_shift: int = 1,
    refresh: bool = False,
) -> pd.Series:
    """
    Per-day sentiment aligned to ``index``; uses JSON cache keyed by ISO date.
    """
    idx = pd.DatetimeIndex(index).normalize()
    cached = {} if refresh else _load_cache(symbol)

    if not cached and os.environ.get("SKIP_EDGAR_SENTIMENT", "").lower() in {"1", "true", "yes"}:
        out = pd.Series(0.0, index=idx, name="finbert_sentiment")
        if pit_shift > 0:
            out = out.shift(pit_shift)
        return out.fillna(0.0)

    if not cached:
        texts = _download_filing_snippets(symbol)
        score = _score_texts(texts)
        for ts in idx:
            cached[ts.strftime("%Y-%m-%d")] = score
        _save_cache(symbol, cached)

    values = [float(cached.get(ts.strftime("%Y-%m-%d"), 0.0)) for ts in idx]
    out = pd.Series(values, index=idx, name="finbert_sentiment")
    if pit_shift > 0:
        out = out.shift(pit_shift)
    return out.fillna(0.0).clip(-1.0, 1.0)
