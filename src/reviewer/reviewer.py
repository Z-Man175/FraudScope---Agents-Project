"""Top-level False-Positive Reviewer orchestrating Code Half, LLM Half, and Hard Vetoes.

WHAT IT DOES
    Evaluates a candidate case produced by the Investigation Agent using three stages:
    1. Code Half (`baselines.py`): Computes strictly past-only (`step < current_step`)
       non-parametric account statistics. 100% memory-blind across ALL setups (B, C, D).
    2. LLM Half (`llm_reviewer.py`): Critically challenges the Investigator's fraud claim
       using bounded Code Half metrics and case summaries. Sees verified memory hints
       ONLY in Setup D.
    3. Hard Veto (`veto_rules.py`): Deterministic safety guardrails that override LLM
       verdicts when Code Half metrics prove the transaction is `HIGHLY_NORMAL` or
       `HIGHLY_ABNORMAL`.

STRICT INVARIANTS
    - Memory firewall: Code Half never reads memory; LLM Half sees memory ONLY in Setup D.
    - 2-round evidence cap: At `review_round >= max_evidence_rounds`, any remaining
      `request_more_evidence` verdict is converted to `"approve"` for escalation.
    - Opinion-vs-truth firewall: Reviewer outputs are system opinions only and NEVER
      populate `final_label`, `label_source`, or `verifier`.
"""

from __future__ import annotations

from typing import Any, Mapping
import pandas as pd

from src.reviewer.baselines import compute_account_baselines
from src.reviewer.llm_reviewer import run_llm_reviewer
from src.reviewer.schemas import (
    EvidenceRequest,
    EvidenceRequestType,
    ReviewerVerdictOutput,
)
from src.reviewer.veto_rules import apply_code_veto


def _as_case_dict(case: Mapping[str, object] | Any) -> dict[str, object]:
    """Convert an `InvestigationCase` dataclass or mapping into a plain dict."""
    if isinstance(case, Mapping):
        return dict(case)
    to_dict_fn = getattr(case, "to_dict", None)
    if callable(to_dict_fn):
        return dict(to_dict_fn())
    raise TypeError(f"Unsupported case type: {type(case)!r}")


def _extract_current_txn(
    case_dict: Mapping[str, object],
    history_df: pd.DataFrame,
) -> dict[str, object]:
    """Extract `step`, `amount`, `type`, `nameOrig`, `nameDest` from a case dict or history."""
    for nested_key in ("current_txn", "transaction", "transaction_summary"):
        nested = case_dict.get(nested_key)
        if isinstance(nested, Mapping) and all(
            k in nested for k in ("step", "amount", "nameOrig", "nameDest")
        ):
            return dict(nested)

    acct_ids = case_dict.get("account_ids", {})
    name_orig = case_dict.get("nameOrig") or (
        acct_ids.get("nameOrig") if isinstance(acct_ids, Mapping) else None
    )
    name_dest = case_dict.get("nameDest") or (
        acct_ids.get("nameDest") if isinstance(acct_ids, Mapping) else None
    )
    step = case_dict.get("step")
    amount = case_dict.get("amount")
    txn_type = case_dict.get("type")

    txn_ids = case_dict.get("transaction_ids")
    primary_txn_id = (
        str(txn_ids[0])
        if isinstance(txn_ids, list) and txn_ids
        else (str(case_dict["txn_id"]) if case_dict.get("txn_id") else None)
    )

    # 1. Try matching the transaction row directly in `history_df` if present
    if (step is None or amount is None) and primary_txn_id and not history_df.empty and "txn_id" in history_df.columns:
        matched = history_df[history_df["txn_id"].astype(str) == primary_txn_id]
        if not matched.empty:
            row = matched.iloc[0]
            if step is None:
                step = int(row["step"])
            if amount is None:
                amount = float(row["amount"])
            if txn_type is None and "type" in matched.columns:
                txn_type = str(row["type"])
            if not name_orig:
                name_orig = str(row["nameOrig"])
            if not name_dest:
                name_dest = str(row["nameDest"])

    # 2. Recover exact candidate `step` from Investigator's `tool_summaries` (`max_step`)
    #    when `history_df` was already pre-filtered to `step < candidate_step` by `pipeline.py`
    if step is None and isinstance(case_dict.get("tool_summaries"), Mapping):
        summaries = case_dict["tool_summaries"]
        for summary_key in ("account_history", "recipient_history", "linked_accounts"):
            sub = summaries.get(summary_key)
            if isinstance(sub, Mapping) and sub.get("max_step") is not None:
                step = int(sub["max_step"])
                break

    # 3. Last-resort fallback if `step` is still unavailable
    if step is None:
        if not history_df.empty and "step" in history_df.columns:
            step = int(history_df["step"].max()) + 1
        else:
            step = 1

    return {
        "txn_id": primary_txn_id or "",
        "step": int(step),
        "amount": float(amount) if amount is not None else 0.0,
        "type": str(txn_type) if txn_type is not None else "TRANSFER",
        "nameOrig": str(name_orig or ""),
        "nameDest": str(name_dest or ""),
    }


def _has_external_chain_flags(case_dict: Mapping[str, object]) -> bool:
    """Return True if the Pattern Agent flagged a multi-hop `account_chain` on this case."""
    detector_scores = case_dict.get("detector_scores", {})
    pattern_types = ""
    if isinstance(detector_scores, Mapping):
        pattern_types = str(detector_scores.get("pattern_types", "") or "")
    if not pattern_types:
        pattern_types = str(case_dict.get("pattern_types", "") or "")
    return "chain" in pattern_types.lower()


def review_case(
    case: dict[str, object] | Any,
    history_df: pd.DataFrame,
    setup: str | None = None,
    review_round: int = 1,
    max_evidence_rounds: int = 2,
    memory_hints: list[dict[str, object]] | None = None,
    peer_df: pd.DataFrame | None = None,
    llm_client: Any = None,
) -> ReviewerVerdictOutput:
    """Review an InvestigationCase using the two-stage Code Half + LLM Half + Hard Veto design."""
    case_dict = _as_case_dict(case)
    case_id = str(case_dict.get("case_id", "CASE_UNKNOWN"))
    effective_round = max(int(review_round), int(case_dict.get("review_round", 1)))
    setup_norm = str(setup).strip().upper() if setup is not None else None

    current_txn = _extract_current_txn(case_dict, history_df)
    chain_flags = _has_external_chain_flags(case_dict)

    # Stage 1: Code Half (strictly past-only step < current_step, 100% memory-blind in all setups)
    code_metrics = compute_account_baselines(
        history_df=history_df,
        current_txn=current_txn,
        peer_df=peer_df,
    )

    # Memory firewall:
    # - When `setup` is explicitly specified ("B", "C", "D"), allow memory ONLY in "D".
    # - When `setup` is None (e.g., called via `pipeline.py` as `review_case(case, past_df)`),
    #   rely on `pipeline.py`'s pre-filtered `case.memory_hints` (empty in B/C, populated in D).
    if setup_norm == "D" or setup_norm is None:
        if memory_hints is not None:
            effective_memory_hints: list[dict[str, object]] | None = list(memory_hints)
        elif isinstance(case_dict.get("memory_hints"), list) and case_dict["memory_hints"]:
            effective_memory_hints = list(case_dict["memory_hints"])  # type: ignore[arg-type]
        else:
            effective_memory_hints = None
    else:
        effective_memory_hints = None

    # Strip any memory_hints from `llm_case_dict` when memory is disabled
    llm_case_dict = dict(case_dict)
    if effective_memory_hints is None:
        llm_case_dict["memory_hints"] = []

    # Stage 2: LLM Half (challenges Investigator claim using Code Half metrics + case summaries)
    llm_result = run_llm_reviewer(
        case_dict=llm_case_dict,
        code_metrics=code_metrics,
        review_round=effective_round,
        memory_hints=effective_memory_hints,
        has_external_chain_flags=chain_flags,
        llm_client=llm_client,
    )
    llm_raw_verdict = str(llm_result["verdict"])
    reviewer_confidence = float(llm_result["reviewer_confidence"])
    rationale = str(llm_result["disproving_rationale"])
    evidence_requests: list[EvidenceRequest] = list(llm_result["evidence_requests"])  # type: ignore[arg-type]

    # Stage 3: Deterministic Code Half Hard Veto (overrides conflicting LLM verdicts)
    final_verdict, veto_triggered, veto_note = apply_code_veto(
        metrics=code_metrics,
        raw_verdict=llm_raw_verdict,
        review_round=effective_round,
        has_external_chain_flags=chain_flags,
    )
    if veto_note:
        rationale = f"{rationale} {veto_note}".strip()

    # Enforce 2-round cap on follow-up evidence requests
    if final_verdict == "request_more_evidence" and effective_round >= int(max_evidence_rounds):
        final_verdict = "approve"
        evidence_requests = []
        rationale = (
            f"{rationale} Evidence round cap reached (round {effective_round} >= "
            f"{max_evidence_rounds}); converting request_more_evidence to 'approve' "
            "for escalation."
        ).strip()
    elif final_verdict == "request_more_evidence" and not evidence_requests:
        # Ensure a round-1 `request_more_evidence` veto always emits a concrete tool request
        default_req_type = (
            EvidenceRequestType.TRACE_DOWNSTREAM_HOPS
            if chain_flags
            else EvidenceRequestType.QUERY_RECIPIENT_FAN_IN
        )
        evidence_requests = [
            EvidenceRequest(
                request_type=default_req_type,
                target_account=current_txn["nameOrig"] or None,
                target_recipient=current_txn["nameDest"] or None,
                window_steps=24,
                reason="Code Half veto requested targeted follow-up evidence in round 1.",
            )
        ]
    elif final_verdict != "request_more_evidence":
        evidence_requests = []

    return ReviewerVerdictOutput(
        case_id=case_id,
        review_round=effective_round,
        code_half=code_metrics,
        llm_raw_verdict=llm_raw_verdict,
        final_verdict=final_verdict,
        code_veto_triggered=veto_triggered,
        reviewer_confidence=round(reviewer_confidence, 4),
        disproving_rationale=rationale,
        evidence_requests=evidence_requests,
    )