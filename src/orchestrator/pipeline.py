"""Plain-Python FraudScope Orchestrator supporting Experimental Setups B, C, and D.

WHAT IT DOES
    Coordinates ranked candidate evaluation across three ablation setups:
    - Setup B (`agents_no_reviewer`): Investigation Agent only (no reviewer loop, no memory).
    - Setup C (`agents_with_reviewer`): Investigation Agent + bounded multi-round Reviewer
      loop (memory strictly disabled so Reviewer relies only on past account history).
    - Setup D (`full_system`): Investigation Agent + Reviewer + verified memory retrieval
      and optional blockchain/hash integrity check (`integrity_checker`).

STRICT INVARIANTS
    1. Label isolation: `isFraud` is stripped from `history_df` before any agent or
       reviewer sees it, and `reviewer_verdict` is NEVER treated as confirmed ground truth.
    2. Confirmed labels: `final_label`, `label_source`, and `verifier` stay `None` unless
       supplied by an explicit external `confirmation_labels` oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import pandas as pd

from src.agents.investigator_agent import (
    InvestigationCase,
    investigate_candidate,
    update_case_with_evidence,
)
from src.orchestrator.llm_client import LLMClient, LLMConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42


class SetupMode(str, Enum):
    B = "B"
    C = "C"
    D = "D"

    @property
    def description(self) -> str:
        return {
            SetupMode.B: "agents_no_reviewer",
            SetupMode.C: "agents_with_reviewer",
            SetupMode.D: "full_system",
        }[self]

    @classmethod
    def parse(cls, value: str | "SetupMode") -> "SetupMode":
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper()
        aliases = {
            "B": cls.B,
            "SETUP_B": cls.B,
            "AGENTS_NO_REVIEWER": cls.B,
            "C": cls.C,
            "SETUP_C": cls.C,
            "AGENTS_WITH_REVIEWER": cls.C,
            "D": cls.D,
            "SETUP_D": cls.D,
            "FULL_SYSTEM": cls.D,
        }
        if norm not in aliases:
            raise ValueError(f"Unsupported setup mode '{value}'. Expected one of B, C, D.")
        return aliases[norm]


@dataclass
class PipelineResult:
    setup: str
    setup_name: str
    records: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    cases_path: Path | None = None
    summary_path: Path | None = None


def _resolve_reviewer(reviewer_fn: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Resolve explicit reviewer callable or auto-import `src.reviewer.review_case`."""
    if reviewer_fn is not None:
        return reviewer_fn
    for module_name in ("src.reviewer", "src.reviewer.reviewer_agent"):
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        fn = getattr(mod, "review_case", None)
        if callable(fn):
            return fn
    return None


def _normalize_reviewer_response(raw: Any) -> dict[str, object]:
    """Normalize dict or dataclass return value from the Reviewer interface."""
    if isinstance(raw, Mapping):
        verdict = raw.get("verdict") or raw.get("reviewer_verdict")
        reason = raw.get("reason") or raw.get("rationale") or ""
        reqs = raw.get("evidence_requests")
        if reqs is None and raw.get("evidence_request") is not None:
            reqs = [raw["evidence_request"]]
        confidence = raw.get("confidence")
    else:
        verdict = getattr(raw, "verdict", None) or getattr(raw, "reviewer_verdict", None)
        reason = getattr(raw, "reason", None) or getattr(raw, "rationale", "") or ""
        reqs = getattr(raw, "evidence_requests", None)
        if reqs is None and getattr(raw, "evidence_request", None) is not None:
            reqs = [getattr(raw, "evidence_request")]
        confidence = getattr(raw, "confidence", None)

    verdict_norm = str(verdict or "").strip().lower()
    if verdict_norm not in {"approve", "reject", "request_more_evidence"}:
        raise ValueError(
            f"Reviewer returned unsupported verdict '{verdict}'. "
            "Expected 'approve', 'reject', or 'request_more_evidence'."
        )

    return {
        "verdict": verdict_norm,
        "reason": str(reason),
        "evidence_requests": list(reqs) if reqs else [],
        "confidence": float(confidence) if confidence is not None else None,
    }


def _resolve_confirmation(
    txn_id: str,
    case_id: str,
    confirmation_labels: Mapping[str, Any] | Callable[..., Any] | None,
) -> tuple[int | None, str | None, str | None]:
    """Resolve optional confirmed ground truth; never derived from reviewer verdict."""
    if confirmation_labels is None:
        return None, None, None

    raw_val: Any = None
    if callable(confirmation_labels):
        raw_val = confirmation_labels(txn_id, case_id)
    elif isinstance(confirmation_labels, Mapping):
        if txn_id in confirmation_labels:
            raw_val = confirmation_labels[txn_id]
        elif case_id in confirmation_labels:
            raw_val = confirmation_labels[case_id]

    if raw_val is None:
        return None, None, None

    if isinstance(raw_val, Mapping):
        lbl = raw_val.get("final_label")
        if lbl is None:
            return None, None, None
        return (
            int(lbl),
            str(raw_val.get("label_source", "external_confirmation")),
            str(raw_val.get("verifier", "confirmation_oracle")),
        )

    return int(raw_val), "external_confirmation", "confirmation_oracle"


def run_pipeline(
    candidates_df: pd.DataFrame,
    history_df: pd.DataFrame,
    setup: str | SetupMode = "C",
    max_k: int | None = None,
    max_evidence_rounds: int = 2,
    llm_client: LLMClient | None = None,
    reviewer_fn: Callable[..., Any] | None = None,
    memory_retriever: Callable[..., Any] | None = None,
    memory_writer: Callable[..., Any] | None = None,
    integrity_checker: Callable[[], bool] | None = None,
    confirmation_labels: Mapping[str, Any] | Callable[..., Any] | None = None,
    out_dir: Path | None = REPO_ROOT / "outputs" / "orchestrator",
    max_retries: int = 2,
) -> PipelineResult:
    """Run the FraudScope investigation & review pipeline across ranked candidates."""
    mode = SetupMode.parse(setup)
    client = llm_client or LLMClient(LLMConfig())
    active_reviewer = _resolve_reviewer(reviewer_fn) if mode in {SetupMode.C, SetupMode.D} else None

    if mode in {SetupMode.C, SetupMode.D} and active_reviewer is None:
        raise ValueError(
            f"Setup {mode.value} requires a reviewer (`reviewer_fn` callable or "
            "`src.reviewer.review_case`)."
        )

    # Strip ground-truth labels so neither the investigator nor reviewer can ever read `isFraud`
    safe_history_df = history_df.drop(columns=["isFraud"], errors="ignore")

    # Setup D optional integrity verification before reading verified memory
    integrity_passed: bool | None = None
    effective_memory_retriever = None
    if mode == SetupMode.D:
        if integrity_checker is not None:
            integrity_passed = bool(integrity_checker())
        if memory_retriever is not None and integrity_passed is not False:
            effective_memory_retriever = memory_retriever

    work_candidates = candidates_df.copy()
    if max_k is not None:
        work_candidates = work_candidates.head(int(max_k))

    records: list[dict[str, object]] = []
    status_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    total_evidence_rounds = 0
    confirmed_count = 0

    for idx, (_, row) in enumerate(work_candidates.iterrows(), start=1):
        cand = row.to_dict()
        step = int(cand["step"])
        txn_id = str(cand.get("txn_id") or f"ROW_{idx:07d}")
        case_id = f"CASE_{txn_id}"

        # Setup C is strictly memory-blind; Setup D queries memory if supplied & valid
        memory_hints: list[dict[str, object]] = []
        if mode == SetupMode.D and effective_memory_retriever is not None:
            retrieved = effective_memory_retriever(
                {
                    "txn_id": txn_id,
                    "step": step,
                    "nameOrig": str(cand["nameOrig"]),
                    "nameDest": str(cand["nameDest"]),
                    "amount": float(cand["amount"]),
                    "pattern_types": str(cand.get("pattern_types", "")),
                }
            )
            if retrieved:
                memory_hints = list(retrieved)

        case: InvestigationCase = investigate_candidate(
            candidate_row=cand,
            history_df=safe_history_df,
            llm_client=client,
            case_id=case_id,
            review_round=1,
            memory_hints=memory_hints if mode == SetupMode.D else None,
            max_retries=max_retries,
        )

        review_history: list[dict[str, object]] = []
        reviewer_verdict: str | None = None
        status = "INVESTIGATED"
        evidence_rounds_used = 0

        if mode in {SetupMode.C, SetupMode.D} and active_reviewer is not None:
            # Strictly past account history (step < candidate step) passed to reviewer
            past_history_df = safe_history_df[safe_history_df["step"] < step]

            rev_resp = _normalize_reviewer_response(active_reviewer(case, past_history_df))
            review_history.append({"round": case.review_round, **rev_resp})
            reviewer_verdict = str(rev_resp["verdict"])

            # Bounded multi-round evidence loop (at most `max_evidence_rounds` follow-ups)
            while reviewer_verdict == "request_more_evidence" and evidence_rounds_used < int(max_evidence_rounds):
                evidence_rounds_used += 1
                reqs = list(rev_resp.get("evidence_requests") or [])
                if not reqs:
                    reqs = [{"request_type": "CHECK_PERIODIC_HISTORY", "payload": {}}]

                case = update_case_with_evidence(
                    case=case,
                    candidate_row=cand,
                    history_df=safe_history_df,
                    evidence_requests=reqs,
                    llm_client=client,
                    memory_retriever=effective_memory_retriever if mode == SetupMode.D else None,
                    max_retries=max_retries,
                )
                rev_resp = _normalize_reviewer_response(active_reviewer(case, past_history_df))
                review_history.append({"round": case.review_round, **rev_resp})
                reviewer_verdict = str(rev_resp["verdict"])

            if reviewer_verdict == "approve":
                status = "APPROVED_FOR_ALERT"
            elif reviewer_verdict == "reject":
                status = "REJECTED_FALSE_POSITIVE"
            else:
                # Still requesting evidence after exhausting max_evidence_rounds -> escalate
                status = "ESCALATED"
                reviewer_verdict = "escalate"

        # Ground-truth confirmation is strictly separate from reviewer_verdict
        final_label, label_source, verifier = _resolve_confirmation(
            txn_id=txn_id,
            case_id=case_id,
            confirmation_labels=confirmation_labels,
        )
        if final_label is not None:
            confirmed_count += 1

        record: dict[str, object] = {
            "case_id": case.case_id,
            "setup": mode.value,
            "txn_id": txn_id,
            "step": step,
            "status": status,
            "reviewer_verdict": reviewer_verdict,
            "evidence_rounds_used": evidence_rounds_used,
            "review_history": review_history,
            "final_label": final_label,
            "label_source": label_source,
            "verifier": verifier,
            "case": case.to_dict(),
        }

        # Only write to verified memory in Setup D when an explicit confirmed label exists
        if mode == SetupMode.D and memory_writer is not None and final_label is not None:
            memory_writer(record)

        records.append(record)
        status_counts[status] = status_counts.get(status, 0) + 1
        verdict_key = str(reviewer_verdict) if reviewer_verdict is not None else "none"
        verdict_counts[verdict_key] = verdict_counts.get(verdict_key, 0) + 1
        total_evidence_rounds += evidence_rounds_used

    n_total = len(records)
    summary: dict[str, object] = {
        "setup": mode.value,
        "setup_name": mode.description,
        "seed": client.config.seed,
        "total_candidates": n_total,
        "max_evidence_rounds": int(max_evidence_rounds),
        "avg_evidence_rounds": round(total_evidence_rounds / n_total, 4) if n_total else 0.0,
        "status_counts": status_counts,
        "verdict_counts": verdict_counts,
        "confirmed_count": confirmed_count,
        "integrity_check_passed": integrity_passed,
        "llm_usage": client.get_usage_summary(),
    }

    cases_path: Path | None = None
    summary_path: Path | None = None
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        cases_path = out_path / f"setup_{mode.value.lower()}_cases.json"
        summary_path = out_path / f"setup_{mode.value.lower()}_summary.json"
        with cases_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return PipelineResult(
        setup=mode.value,
        setup_name=mode.description,
        records=records,
        summary=summary,
        cases_path=cases_path,
        summary_path=summary_path,
    )