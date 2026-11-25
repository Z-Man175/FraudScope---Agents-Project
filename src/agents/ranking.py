"""Joint Anomaly + Pattern Top-K Candidate Ranking for FraudScope

Combines anomaly agent's scores (`outputs/anomaly/{split}_anomaly_scores.parquet`) with
pattern agent's flags (`outputs/pattern/{split}_pattern_flags.parquet`), tunes the blend
weight `alpha` on the validation split, and exports deterministic Top-K candidate tables for
downstream investigation and evaluation

HOW A CANDIDATE IS SCORED
    ranking_score = alpha * normalised_anomaly_score + (1 - alpha) * pattern_score
    alpha = 1.0 is the anomaly-only ranking (the baseline); lower alpha lets pattern flags
    lift a transaction that the anomaly model scored low. Ties are broken by step then txn_id,
    so the same inputs always give the same ranking

THE LABEL COLUMN
    The candidate files keep `isFraud` because the evaluation harness needs it to score the
    run. The investigator and reviewer must NEVER see it. Whatever builds the case file for
    the investigator agent has to drop `isFraud` first (labels only enter as the "confirmation" step)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42

DEFAULT_ALPHA_GRID = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_K_GRID = (25, 50, 100, 200, 500, 1000)

#The default Top-K. It matches CANDIDATE_K in anomaly_agent.py and is a placeholder for the
#real investigation budget. K should come from how many cases the LLM investigator can afford to
#process, NOT from maximising a metric (see tune_ranking_params), so it is fixed unless --tune-k
DEFAULT_TOP_K = 500

#Text columns coming from the pattern table that rank_candidates passes on to the candidates
PATTERN_TEXT_COLS = ("pattern_types", "pattern_reasons", "pattern_event_ids")

FALLBACK_JOIN_COLS = ["step", "nameOrig", "nameDest", "amount"]


def normalize_unit_interval(values: np.ndarray | pd.Series) -> np.ndarray:
    """Normalize numeric scores deterministically to [0.0, 1.0]."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def precision_at_k(y_true: Sequence[int] | np.ndarray | pd.Series, k: int) -> float:
    """Compute Precision@K on labels already ordered from highest to lowest rank."""
    arr = np.asarray(y_true, dtype=int)
    if arr.size == 0 or k <= 0:
        return 0.0
    k_eff = min(int(k), int(arr.size))
    return float(np.sum(arr[:k_eff] == 1) / k_eff)


def recall_at_k(y_true: Sequence[int] | np.ndarray | pd.Series, k: int) -> float:
    """Compute Recall@K on labels already ordered from highest to lowest rank."""
    arr = np.asarray(y_true, dtype=int)
    total_pos = int(np.sum(arr == 1))
    if arr.size == 0 or k <= 0 or total_pos == 0:
        return 0.0
    k_eff = min(int(k), int(arr.size))
    return float(np.sum(arr[:k_eff] == 1) / total_pos)


def f1_at_k(y_true: Sequence[int] | np.ndarray | pd.Series, k: int) -> float:
    """Compute F1@K from Precision@K and Recall@K."""
    p = precision_at_k(y_true, k)
    r = recall_at_k(y_true, k)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def merge_anomaly_and_pattern(
    anomaly_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
) -> pd.DataFrame:
    """Deterministically merge anomaly scores with transaction-level pattern flags

    Uses `txn_id` when available in both tables, otherwise falls back to
    `["step", "nameOrig", "nameDest", "amount"]` with an occurrence index to avoid
    duplicate-key blowups
    """
    anom = anomaly_df.copy()
    pat = pattern_df.copy()

    #Keep only pattern-specific columns on the right side to avoid column collisions
    #Also carries pattern_event_ids, so a candidate can be traced to its pattern events
    pat_extra_cols = [
        c
        for c in ("pattern_score", "pattern_types", "pattern_flag_count", "pattern_reasons", "pattern_event_ids")
        if c in pat.columns
    ]
    if "pattern_score" not in pat_extra_cols:
        raise ValueError("pattern_df must contain a 'pattern_score' column.")

    use_txn_id = "txn_id" in anom.columns and "txn_id" in pat.columns
    if use_txn_id:
        anom["txn_id"] = anom["txn_id"].astype(str)
        pat_right = (
            pat[["txn_id"] + pat_extra_cols]
            .assign(txn_id=lambda d: d["txn_id"].astype(str))
            .drop_duplicates(subset=["txn_id"], keep="first")
        )
        merged = anom.merge(pat_right, on="txn_id", how="left", sort=False)
    else:
        missing_anom = [c for c in FALLBACK_JOIN_COLS if c not in anom.columns]
        missing_pat = [c for c in FALLBACK_JOIN_COLS if c not in pat.columns]
        if missing_anom or missing_pat:
            raise ValueError(
                f"Cannot merge without 'txn_id': missing fallback keys "
                f"(anomaly={missing_anom}, pattern={missing_pat})"
            )
        anom["_dup_idx"] = anom.groupby(FALLBACK_JOIN_COLS, sort=False).cumcount()
        pat_right = pat[FALLBACK_JOIN_COLS + pat_extra_cols].copy()
        pat_right["_dup_idx"] = pat_right.groupby(FALLBACK_JOIN_COLS, sort=False).cumcount()
        merged = anom.merge(
            pat_right,
            on=FALLBACK_JOIN_COLS + ["_dup_idx"],
            how="left",
            sort=False,
        ).drop(columns=["_dup_idx"])

        if "txn_id" not in merged.columns:
            merged.insert(0, "txn_id", [f"ROW_{i:07d}" for i in range(len(merged))])

    merged["pattern_score"] = merged["pattern_score"].fillna(0.0).astype(np.float64).clip(0.0, 1.0)
    #All pattern text columns are filled the same way (before, only pattern_types was)
    #Transactions with no pattern row get empty text and a flag count of 0
    for col in PATTERN_TEXT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
        else:
            merged[col] = ""
    if "pattern_flag_count" in merged.columns:
        merged["pattern_flag_count"] = merged["pattern_flag_count"].fillna(0).astype(np.int64)
    else:
        merged["pattern_flag_count"] = np.int64(0)

    return merged


def score_and_sort_candidates(
    merged_df: pd.DataFrame,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """Compute blended `ranking_score`, sort deterministically, and assign 1-based `rank`."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if "combined_anomaly_score" not in merged_df.columns:
        raise ValueError("merged_df must contain 'combined_anomaly_score'.")

    out = merged_df.copy()
    norm_anom = normalize_unit_interval(out["combined_anomaly_score"])
    pat_score = np.clip(out["pattern_score"].to_numpy(dtype=np.float64), 0.0, 1.0)

    out["norm_anomaly_score"] = norm_anom
    out["ranking_score"] = float(alpha) * norm_anom + (1.0 - float(alpha)) * pat_score

    sort_cols = ["ranking_score", "combined_anomaly_score", "pattern_score", "step", "txn_id"]
    ascending = [False, False, False, True, True]
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    #where this row would rank on the anomaly score alone (1 = most anomalous). Used by
    #rank_candidates to say whether a candidate came from the anomaly side, the pattern side or both
    out["anomaly_rank"] = (
        out["combined_anomaly_score"].rank(method="first", ascending=False).astype(np.int64)
    )
    return out


def rank_candidates(
    anomaly_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    top_k: int = 100,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """Merge anomaly + pattern tables, rank deterministically, and return top-K candidates."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    merged = merge_anomaly_and_pattern(anomaly_df, pattern_df)
    ranked = score_and_sort_candidates(merged, alpha=alpha)
    #`source` says why a row is a candidate. "both" = pattern-flagged AND in the anomaly-only
    #top K; "pattern" = pattern-flagged but outside the anomaly-only top K (the pattern agent
    #rescued it); "anomaly" = not pattern-flagged. Needed to measure what the pattern agent adds
    is_flagged = ranked["pattern_score"] > 0.0
    in_anomaly_top_k = ranked["anomaly_rank"] <= int(top_k)
    ranked["source"] = np.where(
        is_flagged, np.where(in_anomaly_top_k, "both", "pattern"), "anomaly"
    )
    top_df = ranked.head(int(top_k)).copy()

    preferred_cols = [
        "txn_id",
        "step",
        "type",
        "nameOrig",
        "nameDest",
        "amount",
        "isFraud",
        "setup_a_gb_prob",
        "setup_a_pred",
        "iso_anomaly_score",
        "combined_anomaly_score",
        "pattern_score",
        "pattern_types",
        "pattern_flag_count",
        "pattern_reasons",
        "pattern_event_ids",
        "source",
        "ranking_score",
        "rank",
    ]
    ordered_cols = [c for c in preferred_cols if c in top_df.columns]
    return top_df[ordered_cols].reset_index(drop=True)


def _resolve_candidate_ks(n_rows: int, candidate_ks: Sequence[int] | None) -> list[int]:
    if n_rows <= 0:
        return [1]
    base_ks = list(candidate_ks) if candidate_ks is not None else list(DEFAULT_K_GRID)
    valid = sorted({min(int(k), n_rows) for k in base_ks if int(k) > 0})
    if not valid:
        valid = [min(100, n_rows)]
    #If dataset is very small (e.g. unit tests), include finer K values below n_rows
    if n_rows < min(DEFAULT_K_GRID):
        extra = sorted({max(1, int(round(n_rows * frac))) for frac in (0.1, 0.2, 0.3, 0.5)})
        valid = sorted(set(valid) | set(extra))
    return valid


def tune_ranking_params(
    val_anomaly_df: pd.DataFrame,
    val_pattern_df: pd.DataFrame,
    candidate_alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    candidate_ks: Sequence[int] | None = None,
    fixed_alpha: float | None = None,
    fixed_top_k: int | None = None,
) -> dict[str, object]:
    """Tune `alpha` (and optionally `top_k`) on the validation split.

    #CHANGED (documentation only): when `top_k` is tuned, maximising F1@K is a poor way to choose
    #K here. With ~0.13% fraud the F1-best K sits near the number of fraud rows, which is far more
    #than the LLM can investigate, so K is normally FIXED from the budget and only alpha is tuned.
    #Also remember that val is used for tuning, so val metrics are optimistic: report test.

    - When `fixed_top_k` is provided, selects `alpha` maximizing `precision@K` (equivalent
      to `F1@K` at fixed `K`).
    - When `top_k` is also tuned, selects `(alpha, top_k)` maximizing `F1@K` with
      `precision@K` as tie-breaker so the chosen `K` balances precision and recall without
      collapsing trivially to `K=1`.
    """
    merged_val = merge_anomaly_and_pattern(val_anomaly_df, val_pattern_df)
    alphas = [float(fixed_alpha)] if fixed_alpha is not None else [float(a) for a in candidate_alphas]
    ks = (
        [min(int(fixed_top_k), max(len(merged_val), 1))]
        if fixed_top_k is not None
        else _resolve_candidate_ks(len(merged_val), candidate_ks)
    )

    best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    best_result: dict[str, object] = {
        "chosen_alpha": alphas[0],
        "chosen_top_k": ks[0],
        "val_precision_at_k": 0.0,
        "val_recall_at_k": 0.0,
        "val_f1_at_k": 0.0,
    }

    for alpha in alphas:
        ranked = score_and_sort_candidates(merged_val, alpha=alpha)
        y_sorted = ranked["isFraud"].astype(int).to_numpy()
        for k in ks:
            p_k = precision_at_k(y_sorted, k)
            r_k = recall_at_k(y_sorted, k)
            f1_k = f1_at_k(y_sorted, k)
            score_key = (p_k, f1_k, alpha) if fixed_top_k is not None else (f1_k, p_k, alpha)
            if score_key > best_key:
                best_key = score_key
                best_result = {
                    "chosen_alpha": float(alpha),
                    "chosen_top_k": int(k),
                    "val_precision_at_k": float(p_k),
                    "val_recall_at_k": float(r_k),
                    "val_f1_at_k": float(f1_k),
                }

    return best_result


def _split_metrics(ranked_top_df: pd.DataFrame, full_y_true: pd.Series, k: int) -> dict[str, object]:
    n_candidates = int(len(ranked_top_df))
    fraud_in_top_k = int(ranked_top_df["isFraud"].astype(int).sum()) if n_candidates > 0 else 0
    total_fraud = int(full_y_true.astype(int).sum())
    prec = float(fraud_in_top_k / n_candidates) if n_candidates > 0 else 0.0
    rec = float(fraud_in_top_k / total_fraud) if total_fraud > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    return {
        "k": int(k),
        "num_candidates": n_candidates,
        "num_fraud_in_top_k": fraud_in_top_k,
        "total_fraud_in_split": total_fraud,
        "precision_at_k": round(prec, 6),
        "recall_at_k": round(rec, 6),
        "f1_at_k": round(f1, 6),
    }


def run_ranking(
    anomaly_dir: Path = REPO_ROOT / "outputs" / "anomaly",
    pattern_dir: Path = REPO_ROOT / "outputs" / "pattern",
    out_dir: Path = REPO_ROOT / "outputs" / "ranking",
    top_k: int | None = None,
    alpha: float | None = None,
    candidate_alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    candidate_ks: Sequence[int] | None = None,
) -> dict[str, object]:
    """Run joint Anomaly + Pattern Top-K ranking on `val` and `test` and save outputs."""
    anom_path = Path(anomaly_dir)
    pat_path = Path(pattern_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    val_anom = pd.read_parquet(anom_path / "val_anomaly_scores.parquet")
    val_pat = pd.read_parquet(pat_path / "val_pattern_flags.parquet")
    test_anom = pd.read_parquet(anom_path / "test_anomaly_scores.parquet")
    test_pat = pd.read_parquet(pat_path / "test_pattern_flags.parquet")

    tuned = tune_ranking_params(
        val_anomaly_df=val_anom,
        val_pattern_df=val_pat,
        candidate_alphas=candidate_alphas,
        candidate_ks=candidate_ks,
        fixed_alpha=alpha,
        fixed_top_k=top_k,
    )
    chosen_alpha = float(tuned["chosen_alpha"])
    chosen_k = int(tuned["chosen_top_k"])

    summary: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "chosen_top_k": chosen_k,
        "chosen_alpha": chosen_alpha,
        "splits": {},
    }

    for split_name, anom_df, pat_df in [
        ("val", val_anom, val_pat),
        ("test", test_anom, test_pat),
    ]:
        #NOTE: the val numbers printed below are optimistic (alpha was tuned on val); test is the honest one
        top_df = rank_candidates(anom_df, pat_df, top_k=chosen_k, alpha=chosen_alpha)
        candidates_file = out_path / f"{split_name}_top_k_candidates.parquet"
        top_df.to_parquet(candidates_file, index=False)

        metrics = _split_metrics(top_df, anom_df["isFraud"], k=chosen_k)
        summary["splits"][split_name] = metrics
        print(
            f"[{split_name}] Top-{chosen_k} (alpha={chosen_alpha:.2f}) | "
            f"Precision@K={metrics['precision_at_k']:.4f} | "
            f"Recall@K={metrics['recall_at_k']:.4f} | "
            f"F1@K={metrics['f1_at_k']:.4f} | "
            f"Saved {candidates_file} ({len(top_df)} rows)"
        )

    summary_file = out_path / "ranking_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_file}")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Anomaly Agent and Pattern Agent scores into Top-K candidates."
    )
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "anomaly",
        help="Directory containing {split}_anomaly_scores.parquet.",
    )
    parser.add_argument(
        "--pattern-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "pattern",
        help="Directory containing {split}_pattern_flags.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "ranking",
        help="Directory where Top-K candidate parquets and summary JSON will be written.",
    )
    #The default is the fixed budget DEFAULT_TOP_K
    #Pass --tune-k to get the old behaviour (K chosen by F1@K on val)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K investigation budget (default {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--tune-k",
        action="store_true",
        help="Tune K on the val split by F1@K instead of using --top-k.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Explicit anomaly blend weight in [0, 1] (if omitted, tuned on val split).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ranking(
        anomaly_dir=args.anomaly_dir,
        pattern_dir=args.pattern_dir,
        out_dir=args.out_dir,
        top_k=None if args.tune_k else args.top_k,
        alpha=args.alpha,
    )