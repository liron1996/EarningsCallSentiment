"""Shared LLM plumbing for Phase A and Phase B.

Two backends, switchable via the `ECS_LLM_BACKEND` env var:

  ECS_LLM_BACKEND=ollama  (default) - local Ollama daemon
  ECS_LLM_BACKEND=wandb             - W&B Inference (OpenAI-compatible API)

Both backends share the same JSON-repair + <think>-stripping post-processing,
so parse.py and extract.py treat the response identically.

W&B-mode env vars (only required when ECS_LLM_BACKEND=wandb):
  WANDB_API_KEY    your W&B API key
  WANDB_PROJECT    "username_or_team/project_name", e.g. "my_user/earnings-calls"
  ECS_MODEL        a model id from `client.models.list()`, e.g.
                   "meta-llama/Llama-3.1-8B-Instruct"
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from json_repair import repair_json

# Universal knobs (env-overridable).
MODEL = os.environ.get("ECS_MODEL", "qwen3:4b")
NUM_CTX_MAX = int(os.environ.get("ECS_NUM_CTX_MAX", "8192"))
NUM_PREDICT = int(os.environ.get("ECS_NUM_PREDICT", "2048"))
KEEP_ALIVE = os.environ.get("ECS_KEEP_ALIVE", "30m")

# Backend selector.
BACKEND = os.environ.get("ECS_LLM_BACKEND", "ollama").lower()

# Floor used by estimate_num_ctx (kept for API compatibility; we currently always
# return NUM_CTX_MAX so the model stays resident).
NUM_CTX_MIN = 4096

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def estimate_num_ctx(input_chars: int, output_budget: int = NUM_PREDICT) -> int:
    """Always returns NUM_CTX_MAX (kept for API compatibility)."""
    _ = input_chars, output_budget
    return NUM_CTX_MAX


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_json_robust(raw: str) -> Any:
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
    parsed: Any | None
    parse_error: str | None
    model: str


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _call_ollama(
    *,
    system: str,
    user: str,
    num_ctx: int,
    temperature: float,
    seed: int,
    model: str,
) -> tuple[str, str]:
    import ollama  # lazy import so wandb-only users don't need a daemon
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        think=False,
        keep_alive=KEEP_ALIVE,
        options={
            "num_ctx": num_ctx,
            "num_predict": NUM_PREDICT,
            "temperature": temperature,
            "seed": seed,
        },
    )
    return response["message"]["content"], model


# ---------------------------------------------------------------------------
# W&B Inference backend (OpenAI-compatible API)
# ---------------------------------------------------------------------------

_WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
_wandb_client = None


def _get_wandb_client():
    global _wandb_client
    if _wandb_client is not None:
        return _wandb_client
    api_key = os.environ.get("WANDB_API_KEY")
    project = os.environ.get("WANDB_PROJECT")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY env var not set (required for ECS_LLM_BACKEND=wandb)")
    if not project:
        raise RuntimeError(
            "WANDB_PROJECT env var not set (e.g. 'my_user/earnings-calls'). "
            "Required for ECS_LLM_BACKEND=wandb."
        )
    from openai import OpenAI
    _wandb_client = OpenAI(
        base_url=_WANDB_BASE_URL,
        api_key=api_key,
        project=project,
    )
    return _wandb_client


def _call_wandb(
    *,
    system: str,
    user: str,
    temperature: float,
    seed: int,
    model: str,
) -> tuple[str, str]:
    client = _get_wandb_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": NUM_PREDICT,
    }
    use_json_mode = os.environ.get("ECS_USE_JSON_MODE", "1") == "1"
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        # Some models don't support response_format; retry without it.
        if use_json_mode and "response_format" in str(exc).lower():
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    return response.choices[0].message.content or "", model


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def call_llm(
    *,
    system: str,
    user: str,
    num_ctx: int,
    temperature: float = 0.1,
    seed: int = 42,
    model: str = MODEL,
) -> LLMResponse:
    """Call the configured LLM backend; return raw text + best-effort parsed JSON."""
    if BACKEND == "wandb":
        raw_text, model_id = _call_wandb(
            system=system, user=user, temperature=temperature, seed=seed, model=model,
        )
    elif BACKEND == "ollama":
        raw_text, model_id = _call_ollama(
            system=system, user=user, num_ctx=num_ctx,
            temperature=temperature, seed=seed, model=model,
        )
    else:
        raise RuntimeError(
            f"Unknown ECS_LLM_BACKEND={BACKEND!r}. Expected 'ollama' or 'wandb'."
        )

    try:
        parsed = parse_json_robust(raw_text)
        return LLMResponse(text=raw_text, parsed=parsed, parse_error=None, model=model_id)
    except Exception as exc:  # noqa: BLE001
        return LLMResponse(text=raw_text, parsed=None, parse_error=str(exc), model=model_id)


# Back-compat alias - parse.py and extract.py still import this name.
call_ollama = call_llm
