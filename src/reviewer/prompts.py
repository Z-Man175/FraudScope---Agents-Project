"""Prompt templates, canonical JSON serialization, and fixed few-shot examples for the Reviewer.

WHAT IT DOES
    Builds deterministic, reproducible LLM prompts by:
    - Sorting dictionary keys and lists recursively (`_canonicalize`) and rounding floats
      to 2 decimal places (`canonical_json`) so trivial floating-point jitter never alters
      prompt hashes across runs.
    - Providing two anchored few-shot examples (legitimate recurring rent vs. unfamiliar
      burst fraud).
    - Including bounded `tool_summaries` so Round-2 reviews directly inspect any follow-up
      evidence gathered by the Investigation Agent.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.reviewer.schemas import CodeHalfMetrics

PROMPT_VERSION = "reviewer_prompt_v1"

REVIEWER_SYSTEM_PROMPT = (
    "You are the FraudScope False-Positive Reviewer (LLM Half). Your job is to critically "
    "challenge the Investigation Agent's claim using strictly past-only Code Half baseline "
    "metrics and structured tool summaries. Never treat any memory hint as binding over a "
    "HIGHLY_ABNORMAL Code Half baseline, and never emit ground-truth labels. Return ONLY a "
    "valid JSON object with keys: verdict ('approve' | 'reject' | 'request_more_evidence'), "
    "reviewer_confidence (float in [0.0, 1.0]), disproving_rationale (str), and "
    "evidence_requests (list of objects with request_type, target_account, target_recipient, "
    "window_steps, reason)."
)

FEW_SHOT_EXAMPLES: list[dict[str, object]] = [
    {
        "scenario": "recurring_rent_or_landlord_false_positive",
        "input_summary": {
            "normality_band": "HIGHLY_NORMAL",
            "history_status": "ESTABLISHED",
            "amount_percentile": 62.5,
            "robust_z_score": 0.45,
            "is_known_recipient": True,
            "past_tx_to_recipient": 4,
            "periodic_recurrence_flag": True,
            "investigator_claim": "suspicious_requires_review",
        },
        "expected_output": {
            "verdict": "reject",
            "reviewer_confidence": 0.92,
            "disproving_rationale": (
                "Established account history shows recurring periodic transfers of consistent "
                "amount to a known recipient with normal velocity; likely benign rent/payroll."
            ),
            "evidence_requests": [],
        },
    },
    {
        "scenario": "burst_unfamiliar_recipient_fraud",
        "input_summary": {
            "normality_band": "HIGHLY_ABNORMAL",
            "history_status": "ESTABLISHED",
            "amount_percentile": 100.0,
            "max_amount_ratio": 5.2,
            "robust_z_score": 8.4,
            "velocity_1h_ratio": 4.0,
            "is_known_recipient": False,
            "past_tx_to_recipient": 0,
            "periodic_recurrence_flag": False,
            "investigator_claim": "likely_fraud",
        },
        "expected_output": {
            "verdict": "approve",
            "reviewer_confidence": 0.95,
            "disproving_rationale": (
                "Cannot disprove fraud claim: transaction exhibits severe amount and velocity "
                "deviation to a brand-new recipient with no periodic recurrence."
            ),
            "evidence_requests": [],
        },
    },
]


def _canonicalize(obj: Any, ndigits: int = 2) -> Any:
    """Recursively sort keys/lists and round floats for deterministic prompt rendering."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(obj[k], ndigits=ndigits) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple, set)):
        items = [_canonicalize(v, ndigits=ndigits) for v in obj]
        try:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=True))
        except TypeError:
            return items
    return str(obj)


def canonical_json(obj: Any, ndigits: int = 2) -> str:
    """Serialize `obj` deterministically with sorted keys, sorted lists, and rounded floats."""
    return json.dumps(_canonicalize(obj, ndigits=ndigits), sort_keys=True, indent=2)


def build_reviewer_prompt(
    case_payload: Mapping[str, object],
    code_metrics: CodeHalfMetrics,
    review_round: int,
    memory_hints: list[dict[str, object]] | None = None,
) -> str:
    """Construct deterministic reviewer prompt using rounded canonical JSON (no raw tables)."""
    bounded_case: dict[str, object] = {
        "case_id": case_payload.get("case_id", ""),
        "review_round": int(review_round),
        "transaction_ids": case_payload.get("transaction_ids", []),
        "account_ids": case_payload.get("account_ids", {}),
        "detector_scores": case_payload.get("detector_scores", {}),
        "claim": case_payload.get("claim", ""),
        "evidence_for": case_payload.get("evidence_for", []),
        "evidence_against": case_payload.get("evidence_against", []),
        "confidence": case_payload.get("confidence", 0.5),
    }
    # Include bounded tool summaries so Round 2+ can inspect newly requested follow-up evidence
    if isinstance(case_payload.get("tool_summaries"), Mapping) and case_payload["tool_summaries"]:
        bounded_case["tool_summaries"] = dict(case_payload["tool_summaries"])

    prompt_body: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "few_shot_examples": FEW_SHOT_EXAMPLES,
        "code_half_metrics": code_metrics.to_dict(),
        "investigation_case": bounded_case,
    }
    if memory_hints:
        prompt_body["verified_memory_hints_non_binding"] = memory_hints

    round_note = (
        f"This is follow-up review round {review_round}. Weigh the newly gathered tool evidence "
        "in `tool_summaries` and return a final `'approve'` or `'reject'` verdict.\n"
        if int(review_round) > 1
        else ""
    )

    return (
        "Critically evaluate the investigation case against the Code Half past-only baselines. "
        "Return ONLY valid JSON with keys `verdict`, `reviewer_confidence`, "
        "`disproving_rationale`, and `evidence_requests`.\n"
        f"{round_note}\n"
        + canonical_json(prompt_body, ndigits=2)
    )