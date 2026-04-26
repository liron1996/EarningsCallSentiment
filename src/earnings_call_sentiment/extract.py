"""Phase B: LLM-driven sentiment + event extraction.

Pipeline per parsed transcript:
  1. Read data/parsed/<file>.json (output of Phase A).
  2. Format the structured contents as a readable text block for the LLM.
     Include CEO + CFO prepared remarks fully; truncate other prepared blocks if very
     long; cap Q&A at the first ~10 pairs to stay within Qwen3-4B's 8K context.
  3. Call Ollama with extract_v1 prompt -> JSON record (tone, wins, risks, guidance,
     themes, per-speaker tones).
  4. Compute Loughran-McDonald positive/negative word counts on prepared remarks as a
     no-LLM sanity layer; append to the record.
  5. Save to data/extractions/<file>.json. Cache: skip if exists, --force overrides.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import prompts
from ._llm_utils import LLMResponse, call_ollama, estimate_num_ctx

REPO_ROOT = Path(__file__).resolve().parents[2]
PARSED_DIR = REPO_ROOT / "data" / "parsed"
EXTRACTIONS_DIR = REPO_ROOT / "data" / "extractions"
EXTRACTIONS_RAW_DIR = REPO_ROOT / "data" / "extractions_raw"

# Cap individual prepared blocks (mostly to control IR/operator preamble bloat).
MAX_PREPARED_BLOCK_CHARS = 8000
# Cap number of Q&A pairs included (most signal is in the first few; Qwen3-4B context is tight).
MAX_QA_PAIRS = 10
MAX_QA_TEXT_CHARS = 1500  # per question or per answer

# Loughran-McDonald sentiment word lists (compact, hand-curated subset for the sanity layer).
# Full LM dictionary has ~2,000 positive and ~4,000 negative; this subset covers the most
# frequent terms in earnings calls and is good enough for a baseline.
LM_POSITIVE = {
    "achieve", "achieved", "advance", "advantage", "beneficial", "benefit", "best",
    "better", "bolster", "boost", "breakthrough", "compelling", "confidence", "confident",
    "delight", "delivered", "encouraging", "exceed", "exceeded", "exceeding", "excellent",
    "exceptional", "favorable", "gain", "gained", "great", "growth", "healthy", "highest",
    "improve", "improved", "improvement", "leadership", "leading", "milestone", "momentum",
    "outperform", "outstanding", "positive", "progress", "raise", "raised", "record",
    "robust", "solid", "stronger", "strongest", "success", "successful", "surpass",
    "winning", "wins",
}
LM_NEGATIVE = {
    "adverse", "burden", "challenge", "challenged", "challenging", "concern", "concerns",
    "decline", "declined", "declining", "deficit", "delay", "delayed", "deteriorate",
    "deteriorating", "difficult", "difficulty", "disappointing", "disruption", "down",
    "downturn", "fail", "failed", "failure", "fall", "fallen", "headwind", "headwinds",
    "impair", "impairment", "loss", "losses", "lost", "lower", "lowered", "miss", "missed",
    "negative", "pressure", "reduce", "reduced", "shortfall", "slow", "slowdown", "slower",
    "soft", "softer", "softness", "underperform", "uncertain", "uncertainty", "weak",
    "weaken", "weakened", "weaker", "weakness", "worse", "worsening",
}

_WORD_RE = re.compile(r"[A-Za-z']+")


def lm_sentiment(text: str) -> dict[str, float | int]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n = len(words)
    pos = sum(1 for w in words if w in LM_POSITIVE)
    neg = sum(1 for w in words if w in LM_NEGATIVE)
    return {
        "lm_word_count": n,
        "lm_pos": pos,
        "lm_neg": neg,
        "lm_pos_ratio": pos / n if n else 0.0,
        "lm_neg_ratio": neg / n if n else 0.0,
        "lm_net": (pos - neg) / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Format parsed JSON into the user prompt
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " […truncated]"


def format_prepared(prepared_remarks: list[dict]) -> str:
    chunks = []
    for b in prepared_remarks:
        role_norm = b.get("role_norm", "OTHER")
        role_raw = b.get("role_raw", "")
        text = _truncate(b.get("text", ""), MAX_PREPARED_BLOCK_CHARS)
        chunks.append(f"[{role_norm} - {role_raw}]\n{text}")
    return "\n\n".join(chunks) if chunks else "(no prepared remarks parsed)"


def format_qa(qa_pairs: list[dict], max_pairs: int = MAX_QA_PAIRS) -> str:
    if not qa_pairs:
        return "(no Q&A parsed)"
    pieces = []
    truncated = False
    for i, pair in enumerate(qa_pairs[:max_pairs], start=1):
        q = pair.get("question") or {}
        firm = q.get("analyst_firm") or "?"
        q_text = _truncate(q.get("text", ""), MAX_QA_TEXT_CHARS)
        block = [f"--- Q{i} ---", f"ANALYST ({firm}): {q_text}"]
        for ans in pair.get("answers", []):
            role = ans.get("role_norm", "OTHER")
            a_text = _truncate(ans.get("text", ""), MAX_QA_TEXT_CHARS)
            block.append(f"{role}: {a_text}")
        pieces.append("\n".join(block))
    if len(qa_pairs) > max_pairs:
        truncated = True
    out = "\n\n".join(pieces)
    if truncated:
        out += f"\n\n[…{len(qa_pairs) - max_pairs} additional Q&A pair(s) truncated…]"
    return out


def build_user_prompt(parsed: dict) -> tuple[str, str, str]:
    """Returns (full_user_prompt, prepared_text, qa_text). Prepared/qa returned
    separately so the LM sanity layer can score prepared text alone.
    """
    prepared_text = format_prepared(parsed.get("prepared_remarks", []))
    qa_text = format_qa(parsed.get("qa_pairs", []))
    user = prompts.EXTRACT_V1_USER.format(
        ticker=parsed.get("ticker", "?"),
        quarter=parsed.get("quarter", "?"),
        year=parsed.get("year", "?"),
        report_date=parsed.get("report_date", "?"),
        prepared_remarks_block=prepared_text,
        qa_block=qa_text,
    )
    return user, prepared_text, qa_text


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def extract_one(parsed_path: Path, force: bool = False) -> dict[str, Any]:
    out_path = EXTRACTIONS_DIR / parsed_path.name
    if out_path.exists() and not force:
        return {"file": parsed_path.name, "status": "cached"}

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    user_prompt, prepared_text, _qa_text = build_user_prompt(parsed)

    num_ctx = estimate_num_ctx(len(user_prompt))
    response: LLMResponse = call_ollama(
        system=prompts.EXTRACT_V1_SYSTEM,
        user=user_prompt,
        num_ctx=num_ctx,
    )

    if response.parsed is None:
        raw_path = EXTRACTIONS_RAW_DIR / f"{parsed_path.stem}.txt"
        raw_path.write_text(response.text or "<empty>", encoding="utf-8")
        return {
            "file": parsed_path.name,
            "status": "llm_parse_failed",
            "error": response.parse_error,
            "raw_dump": str(raw_path),
        }

    record = {
        "ticker": parsed.get("ticker"),
        "quarter": parsed.get("quarter"),
        "year": parsed.get("year"),
        "report_date": parsed.get("report_date"),
        "filename": parsed.get("filename"),
        **(response.parsed if isinstance(response.parsed, dict) else {"_unexpected_payload": response.parsed}),
        "lm_sentiment": lm_sentiment(prepared_text),
        "_model": response.model,
        "_prompt_version": prompts.EXTRACT_PROMPT_VERSION,
        "_num_ctx": num_ctx,
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"file": parsed_path.name, "status": "ok"}


def extract_batch(files: list[Path], force: bool = False) -> list[dict[str, Any]]:
    results = []
    for p in tqdm(files, desc="extracting"):
        try:
            results.append(extract_one(p, force=force))
        except Exception as exc:  # noqa: BLE001 - never crash batch
            results.append({"file": p.name, "status": "exception", "error": str(exc)})
    return results


def write_summary(results: list[dict[str, Any]]) -> Path:
    summary_path = EXTRACTIONS_DIR / "_summary.json"
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    summary = {"total": len(results), "by_status": by_status, "results": results}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B: LLM-driven sentiment + event extraction")
    parser.add_argument("--files", nargs="*", help="Specific parsed JSON filenames (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args(argv)

    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTIONS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        paths = [PARSED_DIR / f for f in args.files]
    else:
        paths = sorted(p for p in PARSED_DIR.glob("*.json") if not p.name.startswith("_"))

    print(f"Extracting {len(paths)} parsed transcripts (force={args.force})")
    results = extract_batch(paths, force=args.force)
    summary_path = write_summary(results)

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"Done. Status counts: {by_status}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
