"""Strict past-only (`step < current_step`) non-parametric baseline statistics (Code Half).

WHAT IT DOES
    Computes deterministic, distribution-free behavioral baselines for `nameOrig` over
    strictly past transactions (`step < current_step`):
    - History tiering (`ESTABLISHED`, `SPARSE`, `COLD_START`) so cold-start PaySim
      accounts are never falsely cleared by empty personal baselines.
    - Heavy-tail resilient amount statistics (median, MAD-based robust z-score with
      `EPS = 1e-6` protection against zero MAD, max-amount ratio, and peer-group
      percentile fallback for cold-start accounts).
    - Velocity spike ratios (`velocity_1h_ratio`, `velocity_24h_ratio`) compared against
      the account's own 95th-percentile historical cadence.
    - Recipient familiarity (`is_known_recipient`, `past_tx_to_recipient`) and periodic
      cadence detection (weekly `168 +- 12` or monthly `720 +- 24` steps).

STRICT INVARIANT
    Never reads Verified Memory in any setup (B, C, or D).
"""

from __future__ import annotations

import math
from typing import Mapping
import numpy as np
import pandas as pd

from src.reviewer.schemas import CodeHalfMetrics, HistoryTier, NormalityBand
from src.reviewer.veto_rules import classify_normality_band

EPS = 1e-6                      # Guards against division by zero when prior amounts are identical (MAD == 0)
ROLLING_30D_STEPS = 720         # 30 days * 24 hourly steps in PaySim
ESTABLISHED_MIN_TXNS = 10       # Minimum prior transactions to qualify as ESTABLISHED
ESTABLISHED_MIN_SPAN_STEPS = 168  # Minimum 7-day (168-step) span to prevent burst-only accounts qualifying
SPARSE_MIN_TXNS = 3             # 3..9 prior transactions -> SPARSE; 0..2 -> COLD_START


def _is_near_periodic_gap(gap: float) -> bool:
    """Return True if step gap matches weekly (168 +- 12) or monthly (720 +- 24) cadence."""
    return (156.0 <= gap <= 180.0) or (696.0 <= gap <= 744.0)


def _compute_circular_hour_diff(past_steps: np.ndarray, curr_step: int) -> float | None:
    """Compute shortest circular distance (in hours, 0..12) between current hour and usual hour."""
    if len(past_steps) == 0:
        return None
    hours = (past_steps % 24).astype(np.float64)
    angles = hours * (2.0 * math.pi / 24.0)
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))

    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        usual_hour = float(np.median(hours))
    else:
        mean_angle = math.atan2(mean_sin, mean_cos)
        if mean_angle < 0.0:
            mean_angle += 2.0 * math.pi
        usual_hour = mean_angle * (24.0 / (2.0 * math.pi))

    curr_hour = float(int(curr_step) % 24)
    diff = abs(curr_hour - usual_hour) % 24.0
    return round(min(diff, 24.0 - diff), 4)


def _compute_peer_percentile(
    curr_amt: float,
    curr_type: str,
    curr_step: int,
    peer_df: pd.DataFrame | None,
    fallback_past_all: pd.DataFrame,
) -> float | None:
    """Compute same-type peer amount percentile for COLD_START accounts (`step < curr_step`)."""
    if peer_df is not None:
        if peer_df.empty:
            return None
        past_peers = peer_df[peer_df["step"] < curr_step]
    else:
        # `fallback_past_all` is already filtered to `step < curr_step` by caller
        past_peers = fallback_past_all

    if past_peers is None or past_peers.empty:
        return None

    if curr_type and "type" in past_peers.columns:
        typed = past_peers[past_peers["type"] == curr_type]
        if not typed.empty:
            past_peers = typed

    peer_amts = past_peers["amount"].to_numpy(dtype=np.float64)
    if len(peer_amts) == 0:
        return None
    return round(float(100.0 * np.mean(peer_amts <= curr_amt)), 4)


def _compute_velocity_ratios(
    past_steps: np.ndarray,
    curr_step: int,
    history_status: HistoryTier,
) -> tuple[float | None, float | None]:
    """Compare current 1h/24h transaction counts against the account's own 95th percentile."""
    if history_status == HistoryTier.COLD_START or len(past_steps) < SPARSE_MIN_TXNS:
        return None, None

    # Current 1-step (1h) and 24-step (24h) counts including the current transaction
    curr_1h_count = 1 + int(np.sum(past_steps >= curr_step - 1))
    curr_24h_count = 1 + int(np.sum(past_steps >= curr_step - 24))

    # Historical 1-step counts per active step
    _, step_counts = np.unique(past_steps, return_counts=True)
    p95_1h = max(float(np.percentile(step_counts, 95)), 1.0)

    # Historical 24-step rolling counts at each prior transaction step (O(N log N))
    left_indices = np.searchsorted(past_steps, past_steps - 24, side="left")
    right_indices = np.searchsorted(past_steps, past_steps, side="right")
    win24_counts = right_indices - left_indices
    p95_24h = max(float(np.percentile(win24_counts, 95)), 1.0)

    return (
        round(float(curr_1h_count / p95_1h), 4),
        round(float(curr_24h_count / p95_24h), 4),
    )


def compute_account_baselines(
    history_df: pd.DataFrame,
    current_txn: Mapping[str, object],
    peer_df: pd.DataFrame | None = None,
) -> CodeHalfMetrics:
    """Compute strictly past-only (`step < current_txn['step']`) baseline metrics for `nameOrig`."""
    curr_step = int(current_txn["step"])
    curr_orig = str(current_txn["nameOrig"])
    curr_dest = str(current_txn["nameDest"])
    curr_amt = float(current_txn["amount"])
    curr_type = str(current_txn.get("type", "") or "")

    # Strict anti-leakage filter: zero rows with step >= curr_step
    if history_df is None or history_df.empty:
        past_all = pd.DataFrame(columns=["step", "type", "amount", "nameOrig", "nameDest"])
    else:
        past_all = history_df[history_df["step"] < curr_step]

    acct_past = (
        past_all[past_all["nameOrig"].astype(str) == curr_orig]
        .sort_values(["step"], kind="mergesort")
        .reset_index(drop=True)
    )
    recent_30d = acct_past[acct_past["step"] >= curr_step - ROLLING_30D_STEPS]

    past_tx_count = int(len(acct_past))
    past_steps = acct_past["step"].to_numpy(dtype=np.int64) if past_tx_count > 0 else np.array([], dtype=np.int64)
    history_span_steps = int(past_steps[-1] - past_steps[0]) if past_tx_count >= 2 else 0

    # Assign History Tier (prevents cold-start accounts from being treated as established)
    if past_tx_count >= ESTABLISHED_MIN_TXNS and history_span_steps >= ESTABLISHED_MIN_SPAN_STEPS:
        history_status = HistoryTier.ESTABLISHED
        low_history_confidence = False
    elif past_tx_count >= SPARSE_MIN_TXNS:
        history_status = HistoryTier.SPARSE
        low_history_confidence = True
    else:
        history_status = HistoryTier.COLD_START
        low_history_confidence = True

    # Prefer recent 30-day rolling amounts when >= 3 rows exist, otherwise full past history
    ref_df = recent_30d if len(recent_30d) >= SPARSE_MIN_TXNS else acct_past
    ref_amts = ref_df["amount"].to_numpy(dtype=np.float64) if len(ref_df) > 0 else np.array([], dtype=np.float64)
    all_amts = acct_past["amount"].to_numpy(dtype=np.float64) if past_tx_count > 0 else np.array([], dtype=np.float64)

    # Non-parametric scale statistics: MAD-based robust z-score (0.6745 scales MAD to Gaussian sigma)
    if past_tx_count >= 1:
        median_amt = float(np.median(ref_amts))
        mad = float(np.median(np.abs(ref_amts - median_amt)))
        robust_z_score: float | None = round(float(0.6745 * abs(curr_amt - median_amt) / (mad + EPS)), 4)

        max_prior = float(np.max(all_amts))
        max_amount_ratio: float | None = round(float(curr_amt / max_prior), 4) if max_prior > 0.0 else None
    else:
        robust_z_score = None
        max_amount_ratio = None

    if history_status == HistoryTier.COLD_START:
        # Cold-start safe fallback to same-type peer-group percentile
        amount_percentile = _compute_peer_percentile(
            curr_amt=curr_amt,
            curr_type=curr_type,
            curr_step=curr_step,
            peer_df=peer_df,
            fallback_past_all=past_all,
        )
        if amount_percentile is None and len(ref_amts) > 0:
            amount_percentile = round(float(100.0 * np.mean(ref_amts <= curr_amt)), 4)
    else:
        amount_percentile = round(float(100.0 * np.mean(ref_amts <= curr_amt)), 4)

    velocity_1h_ratio, velocity_24h_ratio = _compute_velocity_ratios(
        past_steps=past_steps,
        curr_step=curr_step,
        history_status=history_status,
    )

    if past_tx_count >= 2:
        prior_gaps = np.diff(past_steps)
        curr_gap = curr_step - int(past_steps[-1])
        inter_arrival_percentile: float | None = round(float(100.0 * np.mean(prior_gaps <= curr_gap)), 4)
    else:
        inter_arrival_percentile = None

    # Recipient relationship metrics
    recip_df = acct_past[acct_past["nameDest"].astype(str) == curr_dest]
    past_tx_to_recipient = int(len(recip_df))
    is_known_recipient = past_tx_to_recipient > 0
    total_vol = float(np.sum(all_amts)) if past_tx_count > 0 else 0.0
    recip_vol = float(recip_df["amount"].sum()) if past_tx_to_recipient > 0 else 0.0
    recipient_volume_share = round(float(np.clip(recip_vol / total_vol, 0.0, 1.0)), 4) if total_vol > 0.0 else 0.0

    circular_hour_diff = _compute_circular_hour_diff(past_steps, curr_step)

    # Periodic recurrence check: weekly (168 +- 12) or monthly (720 +- 24) to same recipient or similar amount
    periodic_recurrence_flag = False
    if past_tx_to_recipient > 0:
        recip_gaps = curr_step - recip_df["step"].to_numpy(dtype=np.int64)
        if any(_is_near_periodic_gap(float(g)) for g in recip_gaps):
            periodic_recurrence_flag = True

    if not periodic_recurrence_flag and past_tx_count > 0:
        tol = max(0.15 * max(abs(curr_amt), 1.0), 1.0)
        similar_amt_mask = np.abs(all_amts - curr_amt) <= tol
        if bool(np.any(similar_amt_mask)):
            sim_gaps = curr_step - past_steps[similar_amt_mask]
            if any(_is_near_periodic_gap(float(g)) for g in sim_gaps):
                periodic_recurrence_flag = True

    draft = CodeHalfMetrics(
        history_status=history_status,
        past_tx_count=past_tx_count,
        history_span_steps=history_span_steps,
        low_history_confidence=low_history_confidence,
        amount_percentile=amount_percentile,
        max_amount_ratio=max_amount_ratio,
        robust_z_score=robust_z_score,
        velocity_1h_ratio=velocity_1h_ratio,
        velocity_24h_ratio=velocity_24h_ratio,
        inter_arrival_percentile=inter_arrival_percentile,
        is_known_recipient=is_known_recipient,
        past_tx_to_recipient=past_tx_to_recipient,
        recipient_volume_share=recipient_volume_share,
        circular_hour_diff=circular_hour_diff,
        periodic_recurrence_flag=periodic_recurrence_flag,
        normality_band=NormalityBand.AMBIGUOUS,
        summary_text="",
    )

    band = classify_normality_band(draft)
    pct_str = f"{amount_percentile:.1f}%" if amount_percentile is not None else "N/A"
    z_str = f"{robust_z_score:.2f}" if robust_z_score is not None else "N/A"
    v1_str = f"{velocity_1h_ratio:.2f}x" if velocity_1h_ratio is not None else "N/A"
    v24_str = f"{velocity_24h_ratio:.2f}x" if velocity_24h_ratio is not None else "N/A"

    summary_text = (
        f"Tier={history_status.value} (past_tx={past_tx_count}, span={history_span_steps}h, "
        f"30d_tx={len(recent_30d)}); band={band.value}; amt_pct={pct_str}, robust_z={z_str}, "
        f"vel_1h={v1_str}, vel_24h={v24_str}; known_recipient={is_known_recipient} "
        f"(prior_to_dest={past_tx_to_recipient}); periodic={periodic_recurrence_flag}."
    )

    draft.normality_band = band
    draft.summary_text = summary_text
    return draft