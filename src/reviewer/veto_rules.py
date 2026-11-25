"""Deterministic Code Half normality band classification and hard veto rules.

WHAT IT DOES
    Provides a non-LLM safety layer over the Reviewer LLM's raw verdict:
    1. `classify_normality_band`: Buckets an account's behavior into `HIGHLY_NORMAL`,
       `HIGHLY_ABNORMAL`, or `AMBIGUOUS` based on past-only `CodeHalfMetrics`.
    2. `apply_code_veto`:
       - If `HIGHLY_NORMAL` and LLM says `"approve"`, overrides to `"reject"` (or
         `"request_more_evidence"` in Round 1 when multi-hop chain flags are present).
       - If `HIGHLY_ABNORMAL` and LLM says `"reject"`, overrides to `"request_more_evidence"`
         in Round 1 and `"approve"` in Round 2+.
       - If `AMBIGUOUS` (including `COLD_START` and `SPARSE` tiers), defers to the LLM.

STRICT INVARIANT
    Never reads or depends on Verified Memory in any setup (B, C, or D).
"""

from __future__ import annotations

from src.reviewer.schemas import CodeHalfMetrics, HistoryTier, NormalityBand


def classify_normality_band(metrics: CodeHalfMetrics) -> NormalityBand:
    """Classify an account transaction into `HIGHLY_NORMAL`, `HIGHLY_ABNORMAL`, or `AMBIGUOUS`."""
    # HIGHLY_NORMAL requires an ESTABLISHED account with typical amount, familiar/periodic
    # counterparty behavior, and normal 1h/24h velocity
    is_established = metrics.history_status == HistoryTier.ESTABLISHED
    amount_normal = (
        (metrics.amount_percentile is not None and metrics.amount_percentile <= 85.0)
        or (metrics.robust_z_score is not None and metrics.robust_z_score <= 2.0)
    )
    recipient_or_periodic = (
        metrics.past_tx_to_recipient >= 2 or metrics.periodic_recurrence_flag is True
    )
    vel_1h_normal = metrics.velocity_1h_ratio is None or metrics.velocity_1h_ratio <= 1.2
    vel_24h_normal = metrics.velocity_24h_ratio is None or metrics.velocity_24h_ratio <= 1.2

    if (
        is_established
        and amount_normal
        and recipient_or_periodic
        and vel_1h_normal
        and vel_24h_normal
    ):
        return NormalityBand.HIGHLY_NORMAL

    # HIGHLY_ABNORMAL requires extreme amount (>3.5 sigma AND >3x prior max) OR extreme 1h
    # velocity (>3x p95) sent to an unfamiliar recipient without periodic recurrence
    extreme_amount = (
        (metrics.robust_z_score is not None and metrics.robust_z_score > 3.5)
        and (metrics.max_amount_ratio is not None and metrics.max_amount_ratio > 3.0)
    )
    extreme_velocity = metrics.velocity_1h_ratio is not None and metrics.velocity_1h_ratio > 3.0

    if (
        (extreme_amount or extreme_velocity)
        and (not metrics.is_known_recipient)
        and (not metrics.periodic_recurrence_flag)
    ):
        return NormalityBand.HIGHLY_ABNORMAL

    return NormalityBand.AMBIGUOUS


def apply_code_veto(
    metrics: CodeHalfMetrics,
    raw_verdict: str,
    review_round: int = 1,
    has_external_chain_flags: bool = False,
) -> tuple[str, bool, str]:
    """Apply deterministic Code Half veto against conflicting LLM raw verdicts.

    Returns `(final_verdict, code_veto_triggered, veto_note)`.
    """
    verdict_norm = str(raw_verdict).strip().lower()
    band = classify_normality_band(metrics)

    # Guardrail 1: Prevent false-positive alerts on established routine behavior
    if band == NormalityBand.HIGHLY_NORMAL and verdict_norm == "approve":
        if int(review_round) == 1 and has_external_chain_flags:
            return (
                "request_more_evidence",
                True,
                "Code Half veto (HIGHLY_NORMAL): established routine baseline conflicts with "
                "LLM 'approve' alongside external chain flags; requesting more evidence in round 1.",
            )
        return (
            "reject",
            True,
            "Code Half veto (HIGHLY_NORMAL): established routine baseline overrides LLM 'approve' "
            "to 'reject'.",
        )

    # Guardrail 2: Prevent false negatives on extreme behavioral deviations (and prevent
    # Setup D memory hints from suppressing a severe anomaly)
    if band == NormalityBand.HIGHLY_ABNORMAL and verdict_norm == "reject":
        if int(review_round) == 1:
            return (
                "request_more_evidence",
                True,
                "Code Half veto (HIGHLY_ABNORMAL): severe amount/velocity anomaly to an unfamiliar "
                "recipient overrides LLM 'reject' to 'request_more_evidence' in round 1.",
            )
        return (
            "approve",
            True,
            "Code Half veto (HIGHLY_ABNORMAL): severe amount/velocity anomaly to an unfamiliar "
            "recipient overrides LLM 'reject' to 'approve'.",
        )

    return verdict_norm, False, ""