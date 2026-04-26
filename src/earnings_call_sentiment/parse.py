"""Phase A: LLM-driven transcript parser with chunking + marker-based slicing.

Pipeline per transcript:
  1. Read raw .txt from ECT/.
  2. Chunk the transcript at natural boundaries (Q&A section header, Question/Answer
     block lines). Each chunk's character budget is sized to fit Qwen3-4B's 8K context
     on a 4 GB GPU.
  3. For each chunk, ask the LLM (Ollama) for structured markers (start/end of each
     speaker block).
  4. Slice the original full transcript using those markers - guarantees verbatim
     fidelity even though the LLM only saw a chunk.
  5. Merge per-chunk results: prepared_remarks from prepared chunks; qa_pairs from Q&A
     chunks (linking trailing answers in chunk N to the last question in chunk N).
  6. Dedup adjacent identical blocks (handles JPM_Q1-2026 duplicate intro).
  7. Save JSON to data/parsed/<filename>.json.

Caching: skip if output exists. --force overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import prompts
from ._llm_utils import LLMResponse, NUM_CTX_MAX, call_ollama, estimate_num_ctx

REPO_ROOT = Path(__file__).resolve().parents[2]
ECT_DIR = REPO_ROOT / "ECT"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
PARSED_RAW_DIR = REPO_ROOT / "data" / "parsed_raw"

FILENAME_RE = re.compile(r"^(?P<ticker>[A-Z]+)_Q(?P<quarter>\d)-(?P<year>\d{4})\.txt$")

# Each chunk's input must fit in NUM_CTX_MAX minus the output budget. Local default
# (qwen3:4b on 4 GB VRAM at num_ctx=8192) -> ~14000 chars per chunk. Colab T4 with
# qwen3:14b at num_ctx=32768 -> ~80000 chars per chunk so most transcripts fit in one call.
MAX_CHUNK_CHARS = int(os.environ.get("ECS_MAX_CHUNK_CHARS", "14000"))

# Lines that are safe split points - they always start a new logical block. We include
# "Executives" / "Analysts" / "Operator" role lines so the chunker can break inside the
# Presenter Speech section (which otherwise has no internal section headers).
BLOCK_HEADER_RE = re.compile(
    r"^(Presentation Operator Message|Presenter Speech|"
    r"Question and Answer Operator Message|Question|Answer|"
    r"Executives(?:\s*-\s*.*)?|Analysts(?:\s*-\s*.*)?|Operator(?:\s*-\s*.*)?)$",
    flags=re.MULTILINE,
)


@dataclass
class TranscriptId:
    ticker: str
    quarter: int
    year: int
    filename: str

    @classmethod
    def from_path(cls, path: Path) -> "TranscriptId":
        m = FILENAME_RE.match(path.name)
        if not m:
            raise ValueError(f"Filename does not match TICKER_Q<N>-<YYYY>.txt: {path.name}")
        return cls(
            ticker=m["ticker"],
            quarter=int(m["quarter"]),
            year=int(m["year"]),
            filename=path.name,
        )


@dataclass
class SliceStats:
    marker_misses: int = 0
    duplicates_dropped: int = 0
    orphan_blocks: int = 0
    chunks_used: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_points(text: str) -> list[int]:
    """Character offsets of every line that's a safe block boundary.

    Includes 0 (start of file) and len(text) (end of file) as sentinels.
    """
    points = [0]
    points.extend(m.start() for m in BLOCK_HEADER_RE.finditer(text))
    points.append(len(text))
    return sorted(set(points))


def chunk_transcript(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[int, str]]:
    """Greedy chunker: accumulate sections from one boundary to the next.

    Returns [(offset_in_full_text, chunk_text), ...]. Each chunk is <= max_chars
    where possible (a single section larger than max_chars will exceed it - we
    accept this for the rare case of one giant CEO speech).
    """
    points = _split_points(text)
    chunks: list[tuple[int, str]] = []
    chunk_start = 0
    for i in range(1, len(points)):
        next_end = points[i]
        if next_end - chunk_start > max_chars and points[i - 1] > chunk_start:
            # Emit current chunk ending at the previous boundary.
            chunks.append((chunk_start, text[chunk_start : points[i - 1]]))
            chunk_start = points[i - 1]
    # Trailing chunk.
    if chunk_start < len(text):
        chunks.append((chunk_start, text[chunk_start:]))
    return chunks


# ---------------------------------------------------------------------------
# Marker slicing
# ---------------------------------------------------------------------------

def _coerce_null(value):
    """Some models emit the literal string "null" instead of JSON null."""
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return None
    return value


def find_start_offset(transcript_text: str, block: dict, stats: SliceStats, search_from: int) -> int:
    """Locate block.start_marker in transcript_text starting at search_from.

    Returns -1 on miss (and records a warning); the caller decides how to handle that.
    """
    start_marker = (block.get("start_marker") or "").strip()
    if not start_marker:
        stats.marker_misses += 1
        stats.warnings.append(f"empty_marker role={block.get('role_norm')!r}")
        return -1
    s = transcript_text.find(start_marker, search_from)
    if s == -1:
        # Try a shorter prefix as a fuzzy fallback (LLM may have paraphrased a few chars).
        prefix = start_marker[:30]
        s = transcript_text.find(prefix, search_from) if prefix else -1
        if s == -1:
            stats.marker_misses += 1
            stats.warnings.append(f"start_marker_miss role={block.get('role_norm')!r}")
    return s


def assign_text_from_offsets(blocks: list[dict], offsets: list[int], full_text: str) -> None:
    """Each block's text spans from its start offset to the next block's start offset.

    The last block runs to end of text. Blocks with offset == -1 (marker miss) get the
    marker itself as fallback text. Mutates blocks in place.
    """
    n = len(blocks)
    for i, block in enumerate(blocks):
        if offsets[i] == -1:
            block["text"] = (block.get("start_marker") or "").strip()
            continue
        # Find the next valid offset to use as the end.
        end_offset = len(full_text)
        for j in range(i + 1, n):
            if offsets[j] != -1 and offsets[j] > offsets[i]:
                end_offset = offsets[j]
                break
        block["text"] = full_text[offsets[i] : end_offset].strip()


def dedup_adjacent(blocks: list[dict], stats: SliceStats) -> list[dict]:
    """Drop a block whose text equals the immediately preceding block's text."""
    if not blocks:
        return blocks
    out = [blocks[0]]
    for b in blocks[1:]:
        if b.get("text") and b["text"] == out[-1].get("text"):
            stats.duplicates_dropped += 1
            stats.warnings.append("duplicate_block_dropped")
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Per-chunk LLM call
# ---------------------------------------------------------------------------

def parse_chunk(chunk_text: str) -> LLMResponse:
    num_ctx = estimate_num_ctx(len(chunk_text))
    return call_ollama(
        system=prompts.PARSE_V1_SYSTEM,
        user=prompts.PARSE_V1_USER.format(raw_transcript_text=chunk_text),
        num_ctx=num_ctx,
    )


# ---------------------------------------------------------------------------
# Merge per-chunk results
# ---------------------------------------------------------------------------

def merge_chunks(
    full_text: str,
    chunk_results: list[tuple[int, dict]],
    stats: SliceStats,
) -> tuple[dict, SliceStats]:
    """Walk every block emitted by every chunk, locate each in full_text, then assign
    text by spanning from one start_offset to the next.

    Two-pass design: pass 1 finds offsets for every block (across all chunks), pass 2
    assigns each block's text as the substring up to the next start_offset.
    """
    # Pass 1: collect all blocks in document order, with their start offsets.
    all_blocks: list[dict] = []
    all_offsets: list[int] = []

    prepared_indices: list[int] = []
    qa_meta: list[tuple[int, list[int]]] = []  # (q_idx, [a_idx, ...])
    orphan_indices: list[int] = []

    report_date: str | None = None
    llm_warnings: list[str] = []

    for chunk_offset, llm_json in chunk_results:
        if report_date is None and llm_json.get("report_date"):
            report_date = llm_json["report_date"]
        llm_warnings.extend(llm_json.get("warnings", []))

        for b in llm_json.get("prepared_remarks", []):
            idx = len(all_blocks)
            all_blocks.append(b)
            all_offsets.append(find_start_offset(full_text, b, stats, search_from=chunk_offset))
            prepared_indices.append(idx)

        for pair in llm_json.get("qa_pairs", []):
            q = pair.get("question") or {}
            q["analyst_firm"] = _coerce_null(q.get("analyst_firm"))
            q_idx = len(all_blocks)
            all_blocks.append(q)
            all_offsets.append(find_start_offset(full_text, q, stats, search_from=chunk_offset))
            a_indices: list[int] = []
            for ans in pair.get("answers", []):
                a_idx = len(all_blocks)
                all_blocks.append(ans)
                all_offsets.append(find_start_offset(full_text, ans, stats, search_from=chunk_offset))
                a_indices.append(a_idx)
            qa_meta.append((q_idx, a_indices))

        for b in llm_json.get("orphan_blocks", []):
            idx = len(all_blocks)
            all_blocks.append(b)
            all_offsets.append(find_start_offset(full_text, b, stats, search_from=chunk_offset))
            orphan_indices.append(idx)

    # Pass 2: text spans from one start to the next.
    assign_text_from_offsets(all_blocks, all_offsets, full_text)

    # Re-assemble structures by their indices.
    prepared = dedup_adjacent([all_blocks[i] for i in prepared_indices], stats)
    qa_pairs = [
        {"question": all_blocks[q_idx], "answers": [all_blocks[i] for i in a_idxs]}
        for q_idx, a_idxs in qa_meta
    ]
    orphan_blocks = [all_blocks[i] for i in orphan_indices]

    stats.orphan_blocks = len(orphan_blocks)
    stats.warnings = llm_warnings + stats.warnings

    return {
        "report_date": report_date,
        "prepared_remarks": prepared,
        "qa_pairs": qa_pairs,
        "orphan_blocks": orphan_blocks,
        "warnings": stats.warnings,
    }, stats


# ---------------------------------------------------------------------------
# Top-level per-file driver
# ---------------------------------------------------------------------------

def parse_one(path: Path, force: bool = False) -> dict[str, Any]:
    tid = TranscriptId.from_path(path)
    out_path = PARSED_DIR / f"{tid.ticker}_Q{tid.quarter}-{tid.year}.json"

    if out_path.exists() and not force:
        return {"file": tid.filename, "status": "cached"}

    transcript_text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_transcript(transcript_text)

    stats = SliceStats(chunks_used=len(chunks))
    chunk_results: list[tuple[int, dict]] = []
    chunk_failures: list[dict[str, str]] = []

    for offset, chunk_text in chunks:
        response = parse_chunk(chunk_text)
        if response.parsed is None:
            chunk_failures.append({
                "offset": str(offset),
                "len": str(len(chunk_text)),
                "error": response.parse_error or "<no error>",
            })
            # Persist the raw output so we can debug.
            raw_path = PARSED_RAW_DIR / f"{tid.filename}.chunk{offset}.txt"
            raw_path.write_text(response.text or "<empty>", encoding="utf-8")
            continue
        chunk_results.append((offset, response.parsed))

    if not chunk_results:
        return {
            "file": tid.filename,
            "status": "all_chunks_failed",
            "chunks": stats.chunks_used,
            "failures": chunk_failures,
        }

    sliced, stats = merge_chunks(transcript_text, chunk_results, stats)
    if chunk_failures:
        stats.warnings.append(f"{len(chunk_failures)} chunk(s) failed to parse")

    record = {
        "ticker": tid.ticker,
        "quarter": tid.quarter,
        "year": tid.year,
        "filename": tid.filename,
        **sliced,
        "_model": chunk_results[0][1].get("_model") or "qwen3:4b",
        "_prompt_version": prompts.PARSE_PROMPT_VERSION,
        "_chunks_used": stats.chunks_used,
        "_chunks_failed": len(chunk_failures),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file": tid.filename,
        "status": "ok",
        "chunks": stats.chunks_used,
        "chunks_failed": len(chunk_failures),
        "marker_misses": stats.marker_misses,
        "duplicates_dropped": stats.duplicates_dropped,
        "orphan_blocks": stats.orphan_blocks,
        "warnings_count": len(stats.warnings),
    }


def parse_batch(files: list[Path], force: bool = False) -> list[dict[str, Any]]:
    results = []
    for p in tqdm(files, desc="parsing"):
        try:
            results.append(parse_one(p, force=force))
        except Exception as exc:  # noqa: BLE001 - never crash batch
            results.append({"file": p.name, "status": "exception", "error": str(exc)})
    return results


def write_summary(results: list[dict[str, Any]]) -> Path:
    summary_path = PARSED_DIR / "_summary.json"
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    summary = {"total": len(results), "by_status": by_status, "results": results}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A: LLM-driven earnings-call parser")
    parser.add_argument("--files", nargs="*", help="Specific transcript filenames (default: all in ECT/)")
    parser.add_argument("--force", action="store_true", help="Re-parse even if cached")
    args = parser.parse_args(argv)

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        paths = [ECT_DIR / f for f in args.files]
    else:
        paths = sorted(ECT_DIR.glob("*.txt"))

    print(f"Parsing {len(paths)} transcripts (force={args.force}, MAX_CHUNK_CHARS={MAX_CHUNK_CHARS}, NUM_CTX_MAX={NUM_CTX_MAX})")
    results = parse_batch(paths, force=args.force)
    summary_path = write_summary(results)

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"Done. Status counts: {by_status}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
