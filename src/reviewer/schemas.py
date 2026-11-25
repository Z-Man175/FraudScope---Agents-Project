"""Data schemas and enums for the FraudScope False-Positive Reviewer module.

WHAT IT DOES
    Defines the typed contracts between the Code Half (`CodeHalfMetrics`), LLM Half
    (`EvidenceRequest`), and Orchestrator (`ReviewerVerdictOutput`).

STRICT FIREWALL
    Reviewer outputs represent system opinions only (`approve`, `reject`,
    `request_more_evidence`) and NEVER include ground-truth fields (`final_label`,
    `label_source`, or `verifier`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class HistoryTier(str, Enum):
    """Data-sufficiency tier for the sender account's strictly past history."""

    ESTABLISHED = "ESTABLISHED"  # >= 10 prior txns spanning >= 168 steps (7 days)
    SPARSE = "SPARSE"            # 3..9 prior txns
    COLD_START = "COLD_START"    # 0..2 prior txns


class NormalityBand(str, Enum):
    """Deterministic behavioral classification emitted by the Code Half."""

    HIGHLY_NORMAL = "HIGHLY_NORMAL"
    AMBIGUOUS = "AMBIGUOUS"
    HIGHLY_ABNORMAL = "HIGHLY_ABNORMAL"


class ReviewerVerdictType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"


class EvidenceRequestType(str, Enum):
    """Follow-up tool actions supported by `investigator_agent.fulfill_evidence_request`."""

    CHECK_PERIODIC_HISTORY = "CHECK_PERIODIC_HISTORY"
    QUERY_RECIPIENT_FAN_IN = "QUERY_RECIPIENT_FAN_IN"
    TRACE_DOWNSTREAM_HOPS = "TRACE_DOWNSTREAM_HOPS"
    FETCH_SIMILAR_FALSE_ALARMS = "FETCH_SIMILAR_FALSE_ALARMS"


@dataclass
class EvidenceRequest:
    """Structured follow-up evidence request sent back to the Investigation Agent."""

    request_type: EvidenceRequestType
    target_account: str | None = None
    target_recipient: str | None = None
    window_steps: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        req_str = (
            self.request_type.value
            if isinstance(self.request_type, EvidenceRequestType)
            else str(self.request_type)
        )
        return {
            "request_type": req_str,
            "target_account": self.target_account,
            "target_recipient": self.target_recipient,
            "window_steps": self.window_steps,
            "reason": self.reason,
        }

    def get(self, key: str, default: object = None) -> object:
        """Dict-style compatibility helper for `pipeline.py` and `investigator_agent.py`."""
        if key == "request_type" or key == "type":
            return (
                self.request_type.value
                if isinstance(self.request_type, EvidenceRequestType)
                else str(self.request_type)
            )
        if key == "payload":
            payload: dict[str, object] = {}
            if self.target_account is not None:
                payload["account_id"] = self.target_account
                payload["nameOrig"] = self.target_account
            if self.target_recipient is not None:
                payload["dest_account"] = self.target_recipient
                payload["nameDest"] = self.target_recipient
            if self.window_steps is not None:
                payload["window_steps"] = self.window_steps
            return payload
        return self.to_dict().get(key, default)


@dataclass
class CodeHalfMetrics:
    """Strictly past-only (`step < current_step`) statistical summary of `nameOrig`."""

    history_status: HistoryTier
    past_tx_count: int
    history_span_steps: int
    low_history_confidence: bool
    amount_percentile: float | None
    max_amount_ratio: float | None
    robust_z_score: float | None
    velocity_1h_ratio: float | None
    velocity_24h_ratio: float | None
    inter_arrival_percentile: float | None
    is_known_recipient: bool
    past_tx_to_recipient: int
    recipient_volume_share: float
    circular_hour_diff: float | None
    periodic_recurrence_flag: bool
    normality_band: NormalityBand
    summary_text: str

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["history_status"] = (
            self.history_status.value
            if isinstance(self.history_status, HistoryTier)
            else str(self.history_status)
        )
        raw["normality_band"] = (
            self.normality_band.value
            if isinstance(self.normality_band, NormalityBand)
            else str(self.normality_band)
        )
        return raw


@dataclass
class ReviewerVerdictOutput:
    """Final Reviewer output returned to `src.orchestrator.pipeline`."""

    case_id: str
    review_round: int
    code_half: CodeHalfMetrics
    llm_raw_verdict: str
    final_verdict: str
    code_veto_triggered: bool
    reviewer_confidence: float
    disproving_rationale: str
    evidence_requests: list[EvidenceRequest] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Compatibility alias for `src.orchestrator.pipeline`."""
        return self.final_verdict

    @property
    def reviewer_verdict(self) -> str:
        """Compatibility alias for `src.orchestrator.pipeline`."""
        return self.final_verdict

    @property
    def reason(self) -> str:
        """Compatibility alias for `src.orchestrator.pipeline`."""
        return self.disproving_rationale

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "review_round": int(self.review_round),
            "code_half": self.code_half.to_dict(),
            "llm_raw_verdict": self.llm_raw_verdict,
            "final_verdict": self.final_verdict,
            "code_veto_triggered": bool(self.code_veto_triggered),
            "reviewer_confidence": float(self.reviewer_confidence),
            "disproving_rationale": self.disproving_rationale,
            "evidence_requests": [req.to_dict() for req in self.evidence_requests],
        }