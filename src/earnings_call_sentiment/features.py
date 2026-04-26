"""Phase D: per-call features + QoQ delta features.

Reads:
  - data/extractions/*.json   (Phase B output)
  - data/prices/returns.csv   (Phase C output)

Produces:
  - data/features/features.csv   (one row per call, ML-ready)
  - data/features/_summary.json  (column groups: identifiers / features / labels / debug)

Per call we compute:
  TASK 1 - per-call features (tone, win/risk counts by category, guidance, theme count, LM)
  TASK 2 - QoQ delta features comparing each call to the prior call for the same ticker
           (sentiment delta, count delta, guidance streak, risk persistence, theme drift)

Forward-return labels are left-joined from data/prices/returns.csv on filename.

CLI mirrors parse.py / extract.py: --files filters to specific extraction filenames.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTIONS_DIR = REPO_ROOT / "data" / "extractions"
RETURNS_CSV = REPO_ROOT / "data" / "prices" / "returns.csv"
FEATURES_DIR = REPO_ROOT / "data" / "features"
FEATURES_PATH = FEATURES_DIR / "features.csv"
SUMMARY_PATH = FEATURES_DIR / "_summary.json"

# Schemas come from prompts.py - hardcoded so the column set is stable across runs.
WIN_CATEGORIES = ["revenue", "margin", "product", "customer",
                  "capital_return", "guidance", "operations"]
RISK_CATEGORIES = ["demand", "margin", "competition", "macro", "regulatory",
                   "guidance", "customer_concentration", "supply_chain"]

TONE_BUCKET_ORD = {
    "very_bearish": -2,
    "bearish": -1,
    "neutral": 0,
    "bullish": 1,
    "very_bullish": 2,
}
GUIDANCE_DIRECTION_ORD = {
    "lower": -1,
    "none": 0,
    "reaffirm": 0,
    "raise": 1,
}

RETURN_COLS_LABELS = ["ret_1d", "ret_5d", "ret_21d", "ret_63d",
                      "excess_1d", "excess_5d", "excess_21d", "excess_63d"]
# Pre-call drift columns are FEATURES (input X), not labels - they describe what
# was already in the price before the call happened (look-ahead-safe).
RETURN_COLS_PRECALL = ["pre_5d_ret", "pre_5d_excess",
                       "pre_21d_ret", "pre_21d_excess",
                       "pre_63d_ret", "pre_63d_excess"]
RETURN_COLS_AUX = ["anchor_date", "anchor_close", "pre_anchor_date"]

# Collapse separators so "data_center", "data-center", "datacenter " all map to
# the same surface form. This is a post-hoc normaliser; the v2 prompt also
# instructs the LLM to emit canonical forms in the first place.
_THEME_SEPARATORS_RE = re.compile(r"[_\-/]+")
_THEME_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_theme(s: str) -> str:
    s = s.strip().lower()
    s = _THEME_SEPARATORS_RE.sub(" ", s)
    s = _THEME_WHITESPACE_RE.sub(" ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Task 1: per-call features
# ---------------------------------------------------------------------------

def per_call_features(d: dict) -> dict:
    out: dict[str, Any] = {
        "ticker": d.get("ticker"),
        "quarter": d.get("quarter"),
        "year": d.get("year"),
        "report_date": d.get("report_date"),
        "filename": d.get("filename"),
    }

    out["overall_tone"] = d.get("overall_tone")
    out["ceo_tone"] = d.get("ceo_tone")
    out["cfo_tone"] = d.get("cfo_tone")
    out["qa_tone"] = d.get("qa_tone")
    out["tone_bucket_ord"] = TONE_BUCKET_ORD.get(d.get("tone_bucket"))

    wins = d.get("wins") or []
    win_cats = [w.get("category") for w in wins]
    out["n_wins"] = len(wins)
    for cat in WIN_CATEGORIES:
        out[f"n_wins_{cat}"] = sum(1 for c in win_cats if c == cat)

    risks = d.get("risks") or []
    risk_cats = [r.get("category") for r in risks]
    out["n_risks"] = len(risks)
    for cat in RISK_CATEGORIES:
        out[f"n_risks_{cat}"] = sum(1 for c in risk_cats if c == cat)

    guidance = d.get("guidance") or {}
    out["guidance_direction_ord"] = GUIDANCE_DIRECTION_ORD.get(guidance.get("direction"), 0)
    metrics = set(guidance.get("metrics") or [])
    out["has_revenue_guidance"] = int("revenue" in metrics)
    out["has_margin_guidance"] = int("margin" in metrics)
    out["has_eps_guidance"] = int("eps" in metrics)
    out["has_segment_specific_guidance"] = int("segment_specific" in metrics)
    out["guidance_notes"] = guidance.get("notes") or ""

    themes = [_normalize_theme(t) for t in (d.get("themes") or []) if t]
    themes = [t for t in themes if t]
    # Dedup while preserving order.
    seen: set[str] = set()
    themes = [t for t in themes if not (t in seen or seen.add(t))]
    out["n_themes"] = len(themes)
    out["themes_str"] = ";".join(themes)

    lm = d.get("lm_sentiment") or {}
    out["lm_pos_ratio"] = lm.get("lm_pos_ratio")
    out["lm_neg_ratio"] = lm.get("lm_neg_ratio")
    out["lm_net"] = lm.get("lm_net")
    out["lm_word_count"] = lm.get("lm_word_count")

    return out


# ---------------------------------------------------------------------------
# Task 2: QoQ deltas
# ---------------------------------------------------------------------------

DELTA_NULL: dict[str, Any] = {
    "prev_filename": None,
    "quarters_since_prev": None,
    "d_overall_tone": None,
    "d_ceo_tone": None,
    "d_cfo_tone": None,
    "d_qa_tone": None,
    "d_tone_bucket_ord": None,
    "d_n_wins": None,
    "d_n_risks": None,
    "d_n_themes": None,
    "guidance_streak": None,
    "n_risks_persistent": None,
    "risk_persistence_ratio": None,
    "n_new_themes": None,
    "n_dropped_themes": None,
    "n_persistent_themes": None,
    "new_themes_str": None,
    "d_lm_net": None,
}


def _safe_diff(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return a - b


def compute_deltas(curr: dict, prev: dict | None) -> dict:
    if prev is None:
        return dict(DELTA_NULL)

    out = dict(DELTA_NULL)
    out["prev_filename"] = prev["filename"]
    out["quarters_since_prev"] = (curr["year"] - prev["year"]) * 4 + (curr["quarter"] - prev["quarter"])

    out["d_overall_tone"] = _safe_diff(curr["overall_tone"], prev["overall_tone"])
    out["d_ceo_tone"] = _safe_diff(curr["ceo_tone"], prev["ceo_tone"])
    out["d_cfo_tone"] = _safe_diff(curr["cfo_tone"], prev["cfo_tone"])
    out["d_qa_tone"] = _safe_diff(curr["qa_tone"], prev["qa_tone"])
    out["d_tone_bucket_ord"] = _safe_diff(curr["tone_bucket_ord"], prev["tone_bucket_ord"])

    out["d_n_wins"] = curr["n_wins"] - prev["n_wins"]
    out["d_n_risks"] = curr["n_risks"] - prev["n_risks"]
    out["d_n_themes"] = curr["n_themes"] - prev["n_themes"]

    cd = curr["guidance_direction_ord"]
    pd_ = prev["guidance_direction_ord"]
    out["guidance_streak"] = cd if (cd != 0 and cd == pd_) else 0

    curr_risk_cats = {cat for cat in RISK_CATEGORIES if curr.get(f"n_risks_{cat}", 0) > 0}
    prev_risk_cats = {cat for cat in RISK_CATEGORIES if prev.get(f"n_risks_{cat}", 0) > 0}
    persistent = curr_risk_cats & prev_risk_cats
    out["n_risks_persistent"] = len(persistent)
    denom = max(curr["n_risks"], prev["n_risks"])
    out["risk_persistence_ratio"] = (len(persistent) / denom) if denom > 0 else 0.0

    curr_themes = set(curr["themes_str"].split(";")) if curr["themes_str"] else set()
    prev_themes = set(prev["themes_str"].split(";")) if prev["themes_str"] else set()
    new = curr_themes - prev_themes
    dropped = prev_themes - curr_themes
    persisted = curr_themes & prev_themes
    out["n_new_themes"] = len(new)
    out["n_dropped_themes"] = len(dropped)
    out["n_persistent_themes"] = len(persisted)
    out["new_themes_str"] = ";".join(sorted(new))

    out["d_lm_net"] = _safe_diff(curr["lm_net"], prev["lm_net"])

    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_extractions(filter_files: list[str] | None) -> list[dict]:
    paths = sorted(p for p in EXTRACTIONS_DIR.glob("*.json") if not p.name.startswith("_"))
    if filter_files:
        wanted = set(filter_files)
        paths = [p for p in paths if p.name in wanted]
    out = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        # Override filename with the .json basename so it joins cleanly with
        # data/prices/returns.csv (Phase C also uses the .json basename).
        d["filename"] = p.name
        out.append(d)
    return out


def load_returns() -> pd.DataFrame:
    if not RETURNS_CSV.exists():
        return pd.DataFrame(columns=["filename"])
    return pd.read_csv(RETURNS_CSV)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase D: per-call + QoQ delta features")
    parser.add_argument("--files", nargs="*",
                        help="Extraction filenames to process (default: all)")
    args = parser.parse_args(argv)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    extractions = load_extractions(args.files)
    if not extractions:
        print(f"No extractions found in {EXTRACTIONS_DIR} (filter={args.files})")
        return 1

    rows = [per_call_features(d) for d in extractions]
    # Sort so prior-call lookup walks the timeline forward.
    rows.sort(key=lambda r: (r["ticker"], r["year"] * 4 + r["quarter"]))

    last_per_ticker: dict[str, dict] = {}
    out_rows: list[dict] = []
    counts = {"ok_with_deltas": 0, "ok_no_prior": 0}
    for r in rows:
        prev = last_per_ticker.get(r["ticker"])
        deltas = compute_deltas(r, prev)
        out_rows.append({**r, **deltas})
        counts["ok_with_deltas" if prev is not None else "ok_no_prior"] += 1
        last_per_ticker[r["ticker"]] = r

    df = pd.DataFrame(out_rows)

    # Left-join forward returns AND pre-call drift (both come from Phase C).
    returns = load_returns()
    all_return_cols = RETURN_COLS_AUX + RETURN_COLS_LABELS + RETURN_COLS_PRECALL
    return_cols = [c for c in all_return_cols if c in returns.columns]
    if return_cols:
        df = df.merge(returns[["filename", *return_cols]], on="filename", how="left")
    else:
        for c in all_return_cols:
            df[c] = None

    df.to_csv(FEATURES_PATH, index=False)

    identifiers = ["ticker", "quarter", "year", "report_date", "filename"]
    debug_cols = ["themes_str", "guidance_notes", "new_themes_str", "prev_filename"]
    label_cols = [c for c in RETURN_COLS_LABELS if c in df.columns]
    aux_cols = [c for c in RETURN_COLS_AUX if c in df.columns]
    feature_cols = [c for c in df.columns
                    if c not in identifiers + debug_cols + label_cols + aux_cols]

    summary = {
        "total_rows": len(df),
        "tickers": sorted(df["ticker"].dropna().unique().tolist()),
        "by_status": counts,
        "schema_version": 1,
        "columns": {
            "identifiers": identifiers,
            "features": feature_cols,
            "labels": label_cols,
            "aux": aux_cols,
            "debug": debug_cols,
        },
        "n_features": len(feature_cols),
        "n_labels": len(label_cols),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print(f"Features written to {FEATURES_PATH} ({len(df)} rows x {len(df.columns)} cols)")
    print(f"Status counts: {counts}")
    print(f"Tickers: {summary['tickers']}")
    print(f"Feature columns: {len(feature_cols)} | Label columns: {len(label_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
