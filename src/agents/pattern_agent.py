"""Deterministic Pattern Analysis Agent for FraudScope (no LLM, no randomness)

WHAT IT DOES
    Looks across MANY transactions at once for behaviour that no single row shows. It finds
    three patterns and writes two tables per split (val and test):
        {split}_pattern_events.parquet  one row per detected pattern instance ("event")
        {split}_pattern_flags.parquet   one row per transaction, summarising the events it is in

THE THREE PATTERNS (thresholds live in PatternConfig, the single place to change them)
    burst               >= burst_min_txns transactions from one sender within burst_window_steps
    repeated_recipient  >= repeat_dest_min_txns small transfers into one recipient within
                        repeat_dest_window_steps (small = amount <= small_amt_cutoff)
    account_chain       money forwarded A -> B -> C -> D: each hop leaves within chain_max_step_gap
                        steps of the previous one and forwards chain_amount_ratio_min..max of it,
                        with at least chain_min_hops hops

WHAT IT DELIBERATELY DOES NOT DO
    It does not look at an account's history, so a payroll batch or a rent cycle gets flagged
    exactly like a mule burst. That is intended: telling them apart is the reviewer's job
    (history tiers and baselines). A flag here means "worth a closer look", not "fraud"

HOW THE OUTPUT IS USED
    ranking.py joins {split}_pattern_flags.parquet to the anomaly agent's scores on txn_id
    the investigator agent's evidence tools can look up a flagged transaction's events through `pattern_event_ids`
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42  #recorded in the summary for consistency; nothing in this file is random

REQUIRED_TXN_COLS = {"step", "nameOrig", "nameDest", "amount"}

#The only columns this agent needs from the processed files. Reading just these keeps memory
#low (the features file has ~25 columns we never use)
NEEDED_COLS = ["txn_id", "step", "type", "nameOrig", "nameDest", "amount", "isFraud"]

EVENT_COLS = [
    "event_id",
    "pattern_type",
    "primary_account",
    "involved_accounts",
    "involved_steps",
    "window_start",
    "window_end",
    "txn_ids",
    "pattern_score",
    "reason",
]

#This list lists the real columns of the per-transaction table (type and isFraud are added 
#when present in the input, and pattern_event_ids is new)
TXN_FLAG_COLS = [
    "txn_id",
    "step",
    "type",
    "nameOrig",
    "nameDest",
    "amount",
    "isFraud",
    "is_pattern_flagged",
    "pattern_flag_count",
    "pattern_types",
    "pattern_score",
    "pattern_reasons",
    "pattern_event_ids",
]


#(burst 5 in 6 steps, repeated recipient 4 in 24, chain >= 3 hops within 12 steps at >= 80%)
@dataclass(frozen=True)
class PatternConfig:
    #burst: >= burst_min_txns from one sender whose steps span <= burst_window_steps
    burst_window_steps: int = 6
    burst_min_txns: int = 5
    #repeated recipient: >= repeat_dest_min_txns small transfers into one recipient within the window
    repeat_dest_window_steps: int = 24
    repeat_dest_min_txns: int = 4
    small_amt_quantile: float = 0.5
    #Fixed amount cutoff for "small". None = compute from the data passed in
    #run_pattern_agent fills it from the TRAIN split so val and test use the same number
    small_amt_cutoff: float | None = None
    #chain: hops are transfers, each leaving within chain_max_step_gap steps of the previous one
    #chain_max_hops 3 -> 5 and new chain_min_hops = 3 (before, any 2-hop pass-through counted as
    #a chain and the maximum was 3). chain gap 6 -> 12, ratio min 0.5 -> 0.8
    chain_min_hops: int = 3
    chain_max_hops: int = 5
    chain_max_step_gap: int = 12
    chain_amount_ratio_min: float = 0.8
    chain_amount_ratio_max: float = 1.05  #a hop may forward slightly more than it received (fees, rounding)
    chain_max_branching: int = 5  #cap on how many onward edges are explored per hub (keeps it fast)


#New parameter require_txn_id. The real pipeline always has txn_id (made by Inspect_and_split.py). Silently
#inventing ROW_ ids there would stop the pattern flags joining to the anomaly scores, so run_pattern_agent 
#passes True and fails loudly instead. The ROW_ fallback is kept only for quick experiments and unit tests
def _prepare_transactions(df: pd.DataFrame, require_txn_id: bool = False) -> pd.DataFrame:
    """Validate columns and sort deterministically by (step, txn_id)."""
    missing = REQUIRED_TXN_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input transactions missing required columns: {sorted(missing)}")

    work = df.copy()
    work["_orig_pos"] = np.arange(len(work))

    if "txn_id" in work.columns and work["txn_id"].notna().all():
        work["txn_id"] = work["txn_id"].astype(str)
        work = (
            work.sort_values(["step", "txn_id", "_orig_pos"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        if require_txn_id:
            raise ValueError(
                "Input has no complete 'txn_id' column. Run Inspect_and_split.py first; "
                "inventing ids here would break the join with the anomaly scores."
            )
        work = (
            work.sort_values(["step", "_orig_pos"], kind="mergesort")
            .reset_index(drop=True)
        )
        work["txn_id"] = [f"ROW_{i:07d}" for i in range(len(work))]

    return work.drop(columns=["_orig_pos"])


#Docstring corrected. Spans that overlap are merged; spans that merely sit next to each
#other (end 3, next start 4) are NOT merged, because the test is start <= previous end
def _merge_index_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping [start, end] index spans (touching-but-not-overlapping stay separate)."""
    if not windows:
        return []
    merged: list[tuple[int, int]] = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_sliding_windows(
    steps: np.ndarray,
    max_window_steps: int,
    min_txns: int,
) -> list[tuple[int, int]]:
    """Return merged [left, right] inclusive spans with >= min_txns within max_window_steps.

    `steps` must be sorted. Two pointers: `left` moves forward until the window fits, and every
    window with enough transactions is recorded. Overlapping windows are merged, so a long run of
    activity becomes ONE event instead of many near-duplicates.
    """
    n = len(steps)
    if n < min_txns:
        return []

    raw_windows: list[tuple[int, int]] = []
    left = 0
    for right in range(n):
        while steps[right] - steps[left] > max_window_steps:
            left += 1
        if right - left + 1 >= min_txns:
            raw_windows.append((left, right))

    return _merge_index_windows(raw_windows)


def detect_bursts(df: pd.DataFrame, config: PatternConfig) -> list[dict[str, object]]:
    """Detect rapid bursts of transactions from the same `nameOrig` within `burst_window_steps`."""
    if df.empty:
        return []

    #Cheap pre-filter: an account with fewer than burst_min_txns rows in total cannot burst
    #(In real PaySim most senders appear once, so this removes almost everything)
    counts = df["nameOrig"].value_counts()
    candidate_accounts = sorted(counts[counts >= config.burst_min_txns].index.astype(str))
    if not candidate_accounts:
        return []

    subset = df[df["nameOrig"].isin(candidate_accounts)]
    events: list[dict[str, object]] = []

    for orig, grp in subset.groupby("nameOrig", sort=True):
        steps = grp["step"].to_numpy(dtype=np.int64)
        spans = _find_sliding_windows(
            steps=steps,
            max_window_steps=config.burst_window_steps,
            min_txns=config.burst_min_txns,
        )
        if not spans:
            continue

        dests = grp["nameDest"].astype(str).to_numpy()
        amounts = grp["amount"].to_numpy(dtype=np.float64)
        txn_ids = grp["txn_id"].astype(str).to_numpy()

        for start, end in spans:
            sl = slice(start, end + 1)
            win_steps = steps[sl]
            win_dests = dests[sl]
            win_amts = amounts[sl]
            win_txns = txn_ids[sl]

            count = int(len(win_txns))
            step_span = int(win_steps[-1] - win_steps[0])
            #Score: 0.55 base, +0.08 per extra transaction beyond the minimum, +up to 0.15 when
            #the burst is tight (all in very few steps). Capped at 1.0
            tightness = max(0.0, 1.0 - step_span / max(config.burst_window_steps, 1))
            score = min(
                1.0,
                0.55 + 0.08 * (count - config.burst_min_txns) + 0.15 * tightness,
            )

            involved_accounts = [str(orig)] + sorted(set(win_dests.tolist()) - {str(orig)})
            involved_steps = sorted({int(s) for s in win_steps.tolist()})
            total_amt = float(win_amts.sum())

            events.append(
                {
                    "pattern_type": "burst",
                    "primary_account": str(orig),
                    "involved_accounts": involved_accounts,
                    "involved_steps": involved_steps,
                    "txn_ids": win_txns.tolist(),
                    "pattern_score": round(float(score), 6),
                    "reason": (
                        f"Origin {orig} sent {count} txns within {step_span} steps "
                        f"(steps {involved_steps[0]}..{involved_steps[-1]}, "
                        f"total={total_amt:.2f})"
                    ),
                }
            )

    return events


def detect_repeated_recipient(
    df: pd.DataFrame,
    config: PatternConfig,
) -> list[dict[str, object]]:
    """Detect repeated smaller transfers routed to the same `nameDest` within a step window."""
    if df.empty:
        return []

    if config.small_amt_cutoff is not None:
        small_cutoff = float(config.small_amt_cutoff)
    else:
        small_cutoff = float(df["amount"].quantile(config.small_amt_quantile))
    mask = df["amount"] <= small_cutoff

    #Only money-moving types count; PAYMENT/CASH_IN/DEBIT are ignored (if the type column exists)
    if "type" in df.columns:
        transfer_mask = df["type"].astype(str).isin({"TRANSFER", "CASH_OUT"})
        if bool(transfer_mask.any()):
            mask = mask & transfer_mask

    small_df = df[mask]
    if small_df.empty:
        return []

    counts = small_df["nameDest"].value_counts()
    candidate_dests = sorted(counts[counts >= config.repeat_dest_min_txns].index.astype(str))
    if not candidate_dests:
        return []

    subset = small_df[small_df["nameDest"].isin(candidate_dests)]
    events: list[dict[str, object]] = []

    for dest, grp in subset.groupby("nameDest", sort=True):
        steps = grp["step"].to_numpy(dtype=np.int64)
        spans = _find_sliding_windows(
            steps=steps,
            max_window_steps=config.repeat_dest_window_steps,
            min_txns=config.repeat_dest_min_txns,
        )
        if not spans:
            continue

        origs = grp["nameOrig"].astype(str).to_numpy()
        amounts = grp["amount"].to_numpy(dtype=np.float64)
        txn_ids = grp["txn_id"].astype(str).to_numpy()

        for start, end in spans:
            sl = slice(start, end + 1)
            win_steps = steps[sl]
            win_origs = origs[sl]
            win_amts = amounts[sl]
            win_txns = txn_ids[sl]

            count = int(len(win_txns))
            step_span = int(win_steps[-1] - win_steps[0])
            unique_origs = sorted(set(win_origs.tolist()))
            #Score: 0.50 base, +0.08 per extra txn, +up to 0.12 for several different senders
            #(a collecting/mule account), +up to 0.12 for a tight window. Capped at 1.0
            tightness = max(0.0, 1.0 - step_span / max(config.repeat_dest_window_steps, 1))
            score = min(
                1.0,
                0.50
                + 0.08 * (count - config.repeat_dest_min_txns)
                + 0.12 * min(len(unique_origs), 3) / 3.0
                + 0.12 * tightness,
            )

            involved_accounts = [str(dest)] + [a for a in unique_origs if a != str(dest)]
            involved_steps = sorted({int(s) for s in win_steps.tolist()})
            total_amt = float(win_amts.sum())

            events.append(
                {
                    "pattern_type": "repeated_recipient",
                    "primary_account": str(dest),
                    "involved_accounts": involved_accounts,
                    "involved_steps": involved_steps,
                    "txn_ids": win_txns.tolist(),
                    "pattern_score": round(float(score), 6),
                    "reason": (
                        f"Destination {dest} received {count} small txns (<= {small_cutoff:.2f}) "
                        f"from {len(unique_origs)} origin(s) within {step_span} steps "
                        f"(steps {involved_steps[0]}..{involved_steps[-1]}, total={total_amt:.2f})"
                    ),
                }
            )

    return events


def detect_account_chains(df: pd.DataFrame, config: PatternConfig) -> list[dict[str, object]]:
    """Detect sequential multi-hop fund transfers A -> B -> C [-> D ...] within `chain_max_step_gap`.

    How it works: every transfer is an edge "sender -> recipient at step s for amount a". Starting
    from each edge, a depth-first search follows onward edges that leave the recipient within
    `chain_max_step_gap` steps and forward `chain_amount_ratio_min..max` of the amount. Paths with at
    least `chain_min_hops` hops are kept; paths that are strictly contained in a longer path are dropped.
    """
    #The early exit now compares against chain_min_hops (before: max_hops < 2)
    if df.empty or config.chain_max_hops < config.chain_min_hops:
        return []

    work = df
    if "type" in work.columns:
        transfer_mask = work["type"].astype(str).isin({"TRANSFER", "CASH_OUT"})
        if bool(transfer_mask.any()):
            work = work[transfer_mask]

    if work.empty:
        return []

    #Relay accounts must appear as both a destination (incoming) and an origin (outgoing)
    orig_set = set(work["nameOrig"].astype(str).unique())
    dest_set = set(work["nameDest"].astype(str).unique())
    relay_accounts = orig_set & dest_set
    if not relay_accounts:
        return []

    #Keep only edges touching at least one relay account
    edge_df = work[
        work["nameOrig"].astype(str).isin(relay_accounts)
        | work["nameDest"].astype(str).isin(relay_accounts)
    ]
    if edge_df.empty:
        return []

    #Build deterministic outgoing adjacency indexed by nameOrig (sorted by step, txn_id)
    outgoing_edges: dict[str, list[tuple[int, str, str, float]]] = {}
    outgoing_steps: dict[str, list[int]] = {}

    for row in edge_df[["step", "nameOrig", "nameDest", "amount", "txn_id"]].itertuples(index=False):
        step_val = int(row.step)
        orig = str(row.nameOrig)
        dest = str(row.nameDest)
        amt = float(row.amount)
        tid = str(row.txn_id)
        outgoing_edges.setdefault(orig, []).append((step_val, tid, dest, amt))

    for orig, edges in outgoing_edges.items():
        edges.sort(key=lambda e: (e[0], e[1]))
        outgoing_steps[orig] = [e[0] for e in edges]

    #Collect maximal chains (txn_id tuple -> chain details)
    candidate_chains: list[tuple[list[str], list[int], list[str]]] = []

    def _extend_chain(
        accounts: list[str],
        steps: list[int],
        txn_ids: list[str],
        last_amt: float,
    ) -> None:
        hops = len(txn_ids)
        curr_acct = accounts[-1]
        last_step = steps[-1]
        extended = False

        if hops < config.chain_max_hops and curr_acct in outgoing_edges:
            edges = outgoing_edges[curr_acct]
            edge_steps = outgoing_steps[curr_acct]
            #Binary search for outgoing edges within [last_step, last_step + max_gap]
            lo = bisect_left(edge_steps, last_step)
            hi = bisect_right(edge_steps, last_step + config.chain_max_step_gap)

            matches: list[tuple[int, str, str, float]] = []
            for idx in range(lo, hi):
                nxt_step, nxt_tid, nxt_dest, nxt_amt = edges[idx]
                #Never reuse a transaction or revisit an account (no loops)
                if nxt_tid in txn_ids or nxt_dest in accounts:
                    continue
                if last_amt > 0.0:
                    ratio = nxt_amt / last_amt
                    if not (config.chain_amount_ratio_min <= ratio <= config.chain_amount_ratio_max):
                        continue
                matches.append((nxt_step, nxt_tid, nxt_dest, nxt_amt))

            #Limit branching deterministically if a hub has many matching edges
            for nxt_step, nxt_tid, nxt_dest, nxt_amt in matches[: config.chain_max_branching]:
                extended = True
                _extend_chain(
                    accounts=accounts + [nxt_dest],
                    steps=steps + [nxt_step],
                    txn_ids=txn_ids + [nxt_tid],
                    last_amt=nxt_amt,
                )

        #Keep a path only if it has at least chain_min_hops hops (before: hops >= 2)
        if not extended and hops >= config.chain_min_hops:
            candidate_chains.append((accounts, steps, txn_ids))

    #Start DFS from edges whose destination is a relay account
    start_edges = edge_df[edge_df["nameDest"].astype(str).isin(relay_accounts)]
    for row in start_edges[["step", "nameOrig", "nameDest", "amount", "txn_id"]].itertuples(index=False):
        orig = str(row.nameOrig)
        dest = str(row.nameDest)
        if orig == dest:
            continue
        _extend_chain(
            accounts=[orig, dest],
            steps=[int(row.step)],
            txn_ids=[str(row.txn_id)],
            last_amt=float(row.amount),
        )

    if not candidate_chains:
        return []

    #Deduplicate chains and prune strict sub-chains of longer detected chains
    unique_by_txns: dict[tuple[str, ...], tuple[list[str], list[int], list[str]]] = {}
    for accounts, steps, txn_ids in candidate_chains:
        key = tuple(txn_ids)
        if key not in unique_by_txns:
            unique_by_txns[key] = (accounts, steps, txn_ids)

    all_sets = [(key, set(key)) for key in unique_by_txns]
    maximal_keys: list[tuple[str, ...]] = []
    for key, key_set in all_sets:
        is_sub = any(
            len(other_set) > len(key_set) and key_set.issubset(other_set)
            for other_key, other_set in all_sets
            if other_key != key
        )
        if not is_sub:
            maximal_keys.append(key)

    maximal_keys.sort(key=lambda k: (unique_by_txns[k][1][0], k))

    events: list[dict[str, object]] = []
    for key in maximal_keys:
        accounts, steps, txn_ids = unique_by_txns[key]
        hops = len(txn_ids)
        step_span = int(steps[-1] - steps[0])
        max_possible_span = max((hops - 1) * config.chain_max_step_gap, 1)
        #Score: 0.55 base, +0.15 per extra hop, +up to 0.15 for a tight chain. Capped at 1.0
        tightness = max(0.0, 1.0 - step_span / max_possible_span)
        score = min(1.0, 0.55 + 0.15 * (hops - 1) + 0.15 * tightness)
        involved_steps = sorted({int(s) for s in steps})

        events.append(
            {
                "pattern_type": "account_chain",
                "primary_account": accounts[0],
                "involved_accounts": accounts,
                "involved_steps": involved_steps,
                "txn_ids": txn_ids,
                "pattern_score": round(float(score), 6),
                "reason": (
                    f"Sequential {hops}-hop fund chain {' -> '.join(accounts)} "
                    f"within {step_span} steps (steps {involved_steps[0]}..{involved_steps[-1]})"
                ),
            }
        )

    return events


def aggregate_transaction_flags(
    prepared_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate pattern events to a deterministic per-transaction summary table.

    One row per input transaction (flagged or not). A transaction in several events gets the
    probabilistic OR of their scores, so more independent signals raise the score but never
    past 1.0. Unflagged transactions get pattern_score 0.0.
    """
    base = prepared_df[["txn_id", "step", "nameOrig", "nameDest", "amount"]].copy()
    if "type" in prepared_df.columns:
        base["type"] = prepared_df["type"].values
    if "isFraud" in prepared_df.columns:
        base["isFraud"] = prepared_df["isFraud"].values

    if events_df.empty:
        base["is_pattern_flagged"] = False
        base["pattern_flag_count"] = np.int64(0)
        base["pattern_types"] = ""
        base["pattern_score"] = np.float64(0.0)
        base["pattern_reasons"] = ""
        base["pattern_event_ids"] = ""
        return base

    #Map each txn_id to all pattern events touching it
    txn_types: dict[str, set[str]] = {}
    txn_scores: dict[str, list[float]] = {}
    txn_reasons: dict[str, list[str]] = {}
    txn_event_ids: dict[str, list[str]] = {}

    #event_id is read too, so each transaction remembers which events it belongs to
    for row in events_df[["event_id", "pattern_type", "txn_ids", "pattern_score", "reason"]].itertuples(index=False):
        ptype = str(row.pattern_type)
        score = float(np.clip(row.pattern_score, 0.0, 1.0))
        reason = str(row.reason)
        for tid in row.txn_ids:
            tid_str = str(tid)
            txn_types.setdefault(tid_str, set()).add(ptype)
            txn_scores.setdefault(tid_str, []).append(score)
            txn_reasons.setdefault(tid_str, []).append(reason)
            txn_event_ids.setdefault(tid_str, []).append(str(row.event_id))

    is_flagged: list[bool] = []
    flag_counts: list[int] = []
    type_strs: list[str] = []
    agg_scores: list[float] = []
    reason_strs: list[str] = []
    event_id_strs: list[str] = []

    for tid in base["txn_id"].astype(str):
        scores = txn_scores.get(tid)
        if not scores:
            is_flagged.append(False)
            flag_counts.append(0)
            type_strs.append("")
            agg_scores.append(0.0)
            reason_strs.append("")
            event_id_strs.append("")
        else:
            #Probabilistic OR smoothly combines multiple pattern signals within [0.0, 1.0]
            combined = 1.0 - float(np.prod([1.0 - s for s in scores]))
            is_flagged.append(True)
            flag_counts.append(len(scores))
            type_strs.append(",".join(sorted(txn_types[tid])))
            agg_scores.append(round(float(np.clip(combined, 0.0, 1.0)), 6))
            reason_strs.append(" | ".join(sorted(set(txn_reasons[tid]))))
            event_id_strs.append(",".join(sorted(set(txn_event_ids[tid]))))

    base["is_pattern_flagged"] = np.asarray(is_flagged, dtype=bool)
    base["pattern_flag_count"] = np.asarray(flag_counts, dtype=np.int64)
    base["pattern_types"] = type_strs
    base["pattern_score"] = np.asarray(agg_scores, dtype=np.float64)
    base["pattern_reasons"] = reason_strs
    base["pattern_event_ids"] = event_id_strs
    return base


def detect_patterns(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
    require_txn_id: bool = False,
    id_prefix: str = "EV",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all deterministic pattern detectors on `df`.

    Returns:
        events_df: event-level pattern flags with `EVENT_COLS`.
        pattern_flags_df: transaction-level summary table for ranking and investigation.
    """
    cfg = config or PatternConfig()
    prepared = _prepare_transactions(df, require_txn_id=require_txn_id)

    events: list[dict[str, object]] = []
    events.extend(detect_bursts(prepared, cfg))
    events.extend(detect_repeated_recipient(prepared, cfg))
    events.extend(detect_account_chains(prepared, cfg))

    if events:
        for ev in events:
            ev["window_start"] = int(ev["involved_steps"][0])
            ev["window_end"] = int(ev["involved_steps"][-1])
        #event_id is added below, so build the frame from every column except event_id
        events_df = pd.DataFrame(events, columns=[c for c in EVENT_COLS if c != "event_id"])
        events_df = (
            events_df.sort_values(
                ["pattern_type", "primary_account", "pattern_score"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        #ids are assigned AFTER the deterministic sort, so the same input always gives the
        #same ids (EV_VAL_00001, EV_VAL_00002, ...)
        events_df.insert(0, "event_id", [f"{id_prefix}_{i + 1:05d}" for i in range(len(events_df))])
    else:
        events_df = pd.DataFrame(columns=EVENT_COLS)

    pattern_flags_df = aggregate_transaction_flags(prepared, events_df)
    return events_df, pattern_flags_df


#Reads raw_clean first (smaller than features) and only the NEEDED_COLS. Before, it
#loaded every column of the features file for 6M-row splits
def _load_split_transactions(processed_dir: Path, split: str) -> pd.DataFrame:
    for name in (f"{split}_raw_clean.parquet", f"{split}_features.parquet"):
        path = processed_dir / name
        if path.exists():
            available = set(pq.read_schema(path).names)
            return pd.read_parquet(path, columns=[c for c in NEEDED_COLS if c in available])
    raise FileNotFoundError(
        f"Could not find {split}_raw_clean.parquet or {split}_features.parquet in {processed_dir}."
    )


#The "small amount" cutoff is computed ONCE from the train split and reused for every split
#Using train keeps it leak-free and makes val and test comparable
def _train_small_amount_cutoff(processed_dir: Path, quantile: float) -> float | None:
    for name in ("train_raw_clean.parquet", "train_features.parquet"):
        path = processed_dir / name
        if path.exists():
            amounts = pd.read_parquet(path, columns=["amount"])["amount"]
            return float(amounts.quantile(quantile))
    return None


def run_pattern_agent(
    processed_dir: Path = REPO_ROOT / "data" / "processed",
    out_dir: Path = REPO_ROOT / "outputs" / "pattern",
    splits: Iterable[str] = ("val", "test"),
    config: PatternConfig | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run the Pattern Analysis Agent across splits and persist parquet + JSON outputs."""
    processed_path = Path(processed_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = config or PatternConfig()

    #Fill in the fixed amount cutoff from train (unless the caller already set one)
    if cfg.small_amt_cutoff is None:
        train_cutoff = _train_small_amount_cutoff(processed_path, cfg.small_amt_quantile)
        if train_cutoff is not None:
            cfg = replace(cfg, small_amt_cutoff=train_cutoff)

    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    summary: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "config": asdict(cfg),
        "splits": {},
    }

    for split in splits:
        df_split = _load_split_transactions(processed_path, split)
        #require_txn_id=True and a per-split event id prefix
        events_df, flags_df = detect_patterns(
            df_split, cfg, require_txn_id=True, id_prefix=f"EV_{split.upper()}"
        )

        flags_file = out_path / f"{split}_pattern_flags.parquet"
        events_file = out_path / f"{split}_pattern_events.parquet"
        flags_df.to_parquet(flags_file, index=False)
        events_df.to_parquet(events_file, index=False)

        counts_by_type = (
            events_df["pattern_type"].value_counts().sort_index().to_dict()
            if not events_df.empty
            else {}
        )
        n_flagged_txns = int(flags_df["is_pattern_flagged"].sum())
        summary["splits"][split] = {
            "total_transactions": int(len(flags_df)),
            "flagged_transactions": n_flagged_txns,
            "total_pattern_events": int(len(events_df)),
            "events_by_type": {str(k): int(v) for k, v in counts_by_type.items()},
        }

        #How well do the flags line up with the labels? Evaluation only. The labels play no
        #part in detection. Read these numbers before trusting the thresholds: a very low
        #flagged_precision means the rules fire mostly on normal rows
        if "isFraud" in flags_df.columns:
            is_fraud = flags_df["isFraud"].astype(int) == 1
            total_fraud = int(is_fraud.sum())
            flagged_fraud = int((is_fraud & flags_df["is_pattern_flagged"]).sum())
            summary["splits"][split].update(
                {
                    "total_fraud": total_fraud,
                    "flagged_fraud": flagged_fraud,
                    "flag_recall": round(flagged_fraud / total_fraud, 6) if total_fraud else 0.0,
                    "flagged_precision": (
                        round(flagged_fraud / n_flagged_txns, 6) if n_flagged_txns else 0.0
                    ),
                }
            )

        print(
            f"[{split}] Saved {flags_file} ({len(flags_df)} rows, "
            f"{n_flagged_txns} flagged) | events={counts_by_type}"
        )
        results[split] = (events_df, flags_df)

    summary_file = out_path / "pattern_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_file}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic FraudScope Pattern Agent.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory containing {split}_raw_clean.parquet or {split}_features.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "pattern",
        help="Directory where pattern flag outputs will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Dataset splits to analyze.",
    )
    #The command-line defaults now come from PatternConfig, so they cannot drift apart
    d = PatternConfig()
    parser.add_argument("--burst-window-steps", type=int, default=d.burst_window_steps)
    parser.add_argument("--burst-min-txns", type=int, default=d.burst_min_txns)
    parser.add_argument("--repeat-dest-window-steps", type=int, default=d.repeat_dest_window_steps)
    parser.add_argument("--repeat-dest-min-txns", type=int, default=d.repeat_dest_min_txns)
    parser.add_argument("--small-amt-quantile", type=float, default=d.small_amt_quantile)
    parser.add_argument("--small-amt-cutoff", type=float, default=None,
                        help="Fixed amount cutoff for 'small' (default: quantile of the train split).")  #NEW
    parser.add_argument("--chain-min-hops", type=int, default=d.chain_min_hops)  #NEW
    parser.add_argument("--chain-max-hops", type=int, default=d.chain_max_hops)
    parser.add_argument("--chain-max-step-gap", type=int, default=d.chain_max_step_gap)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = PatternConfig(
        burst_window_steps=args.burst_window_steps,
        burst_min_txns=args.burst_min_txns,
        repeat_dest_window_steps=args.repeat_dest_window_steps,
        repeat_dest_min_txns=args.repeat_dest_min_txns,
        small_amt_quantile=args.small_amt_quantile,
        small_amt_cutoff=args.small_amt_cutoff,
        chain_min_hops=args.chain_min_hops,
        chain_max_hops=args.chain_max_hops,
        chain_max_step_gap=args.chain_max_step_gap,
    )
    run_pattern_agent(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        splits=args.splits,
        config=cfg,
    )