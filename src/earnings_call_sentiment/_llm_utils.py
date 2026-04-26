"""Shared LLM plumbing for Phase A and Phase B.

Centralises Ollama client construction, num_ctx sizing, <think> stripping, and
JSON repair. Both parse.py and extract.py use this module so failure handling
behaves identically.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import ollama
from json_repair import repair_json

# All knobs are env-var overridable so the same code runs on local 4 GB hardware
# and on Colab T4 (15 GB VRAM) without code changes. Defaults target the local box;
# Colab notebook bumps them via os.environ before importing.
MODEL = os.environ.get("ECS_MODEL", "qwen3:4b")
NUM_CTX_MAX = int(os.environ.get("ECS_NUM_CTX_MAX", "8192"))
NUM_PREDICT = int(os.environ.get("ECS_NUM_PREDICT", "2048"))
KEEP_ALIVE = os.environ.get("ECS_KEEP_ALIVE", "30m")

# Floor used by estimate_num_ctx (kept for API compatibility; we currently always
# return NUM_CTX_MAX so the model stays resident).
NUM_CTX_MIN = 4096

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def estimate_num_ctx(input_chars: int, output_budget: int = NUM_PREDICT) -> int:
    """ALWAYS returns NUM_CTX_MAX.

    We previously sized num_ctx per call. That triggered Ollama to reload the model
    each time num_ctx changed, which on a memory-constrained machine added 30-60s of
    overhead per chunk. Using a single fixed num_ctx keeps the model resident across
    every call. The (input_chars, output_budget) args are kept for API compatibility
    and future tuning.
    """
    _ = input_chars, output_budget  # intentionally unused
    return NUM_CTX_MAX


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks. Defensive even when think=False is set."""
    return _THINK_RE.sub("", text).strip()


def parse_json_robust(raw: str) -> Any:
    """Try strict parse, then json-repair, then raise.

    Caller is responsible for catching the exception and dumping `raw` to disk
    so the failure can be debugged.
    """
    cleaned = strip_think(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    repaired = repair_json(cleaned)
    return json.loads(repaired)


@dataclass
class LLMResponse:
    text: str
    parsed: Any | None  # None if parsing failed
    parse_error: str | None
    model: str


def call_ollama(
    *,
    system: str,
    user: str,
    num_ctx: int,
    temperature: float = 0.1,
    seed: int = 42,
    model: str = MODEL,
) -> LLMResponse:
    """Call Ollama chat with our standard settings (think off, JSON format, fixed seed).

    Returns the raw text plus a best-effort parsed JSON value. If parsing fails the
    `parsed` field is None and `parse_error` carries the exception message; the caller
    should persist `text` to disk for debugging.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        think=False,
        keep_alive=KEEP_ALIVE,  # keep model loaded between calls (vs. default 5m)
        options={
            "num_ctx": num_ctx,
            "num_predict": NUM_PREDICT,
            "temperature": temperature,
            "seed": seed,
        },
    )
    raw_text = response["message"]["content"]

    try:
        parsed = parse_json_robust(raw_text)
        return LLMResponse(text=raw_text, parsed=parsed, parse_error=None, model=model)
    except Exception as exc:  # noqa: BLE001 - we want any parse failure to surface in the field
        return LLMResponse(text=raw_text, parsed=None, parse_error=str(exc), model=model)
