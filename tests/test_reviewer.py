"""Unit tests for Freedy's False-Positive Reviewer module (`src/reviewer/`)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.llm_client import LLMClient, LLMConfig, LLMResponse
from src.reviewer import (
    CodeHalfMetrics,
    HistoryTier,
    NormalityBand,
    ReviewerVerdictOutput,
    apply_code_veto,
    classify_normality_band,
    compute_account_baselines,
    review_case,
)


def _make_established_history() -> pd.DataFrame:
    """12 prior weekly/spaced transactions for O_EST spanning > 168 steps."""
    rows = []
    # 12 transactions across steps 10..420 (span = 410 >= 168)
    for i in range(12):
        step = 10 + i * 35  # 10, 45, 80, ..., 395
        dest = "D_KNOWN" if i % 2 == 0 else f"D_OTHER_{i}"
        rows.append(
            {
                "txn_id": f"TXN_EST_{i:03d}",
                "step": step,
                "type": "TRANSFER",
                "amount": 1_000.0 + (i % 3) * 50.0,
                "nameOrig": "O_EST",
                "nameDest": dest,
                "isFraud": 0,
            }
        )
    # Add a specific weekly anchor 168 steps before step 500 -> step 332 to D_KNOWN
    rows.append(
        {
            "txn_id": "TXN_EST_WEEKLY",
            "step": 332,
            "type": "TRANSFER",
            "amount": 1_050.0,
            "nameOrig": "O_EST",
            "nameDest": "D_KNOWN",
            "isFraud": 0,
        }
    )
    return pd.DataFrame(rows)


class _ScriptedReviewerLLM(LLMClient):
    """Mock LLMClient that captures prompts and returns scripted JSON responses."""

    def __init__(self, responses: list[str], log_path: Path) -> None:
        super().__init__(LLMConfig(log_path=log_path))
        self._responses = list(responses)
        self.captured_prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: dict | None = None,
        case_id: str | None = None,
        step_name: str = "unknown",
    ) -> LLMResponse:
        self.captured_prompts.append(prompt)
        resp = super().generate(prompt, system_prompt, schema_hint, case_id, step_name)
        idx = min(len(self.captured_prompts) - 1, len(self._responses) - 1)
        resp.content = self._responses[idx]
        resp.mode = "mock_llm"
        return resp


def test_baselines_strict_past_filtering_excludes_future_rows() -> None:
    past_df = _make_established_history()
    future_rows = pd.DataFrame(
        [
            # Same step == 500 and future step == 505 must be completely ignored!
            {"txn_id": "TXN_CURR", "step": 500, "type": "TRANSFER", "amount": 999_999.0, "nameOrig": "O_EST", "nameDest": "D_NEW", "isFraud": 1},
            {"txn_id": "TXN_FUT1", "step": 505, "type": "TRANSFER", "amount": 999_999.0, "nameOrig": "O_EST", "nameDest": "D_NEW", "isFraud": 1},
            {"txn_id": "TXN_FUT2", "step": 506, "type": "TRANSFER", "amount": 999_999.0, "nameOrig": "O_EST", "nameDest": "D_NEW", "isFraud": 1},
        ]
    )
    combined = pd.concat([past_df, future_rows], ignore_index=True)
    current_txn = {
        "step": 500,
        "type": "TRANSFER",
        "amount": 1_050.0,
        "nameOrig": "O_EST",
        "nameDest": "D_KNOWN",
    }

    m_past_only = compute_account_baselines(past_df, current_txn)
    m_with_future = compute_account_baselines(combined, current_txn)

    assert m_past_only.to_dict() == m_with_future.to_dict()
    assert m_with_future.past_tx_count == len(past_df)
    assert m_with_future.history_status == HistoryTier.ESTABLISHED
    assert m_with_future.periodic_recurrence_flag is True
    assert m_with_future.normality_band == NormalityBand.HIGHLY_NORMAL


def test_history_tiers_and_zero_mad_cold_start_safety() -> None:
    # 1. ESTABLISHED with identical prior amounts (MAD == 0.0 -> no division by zero!)
    const_df = pd.DataFrame(
        [
            {"step": 10 + i * 24, "type": "TRANSFER", "amount": 500.0, "nameOrig": "O_CONST", "nameDest": "D_RENT"}
            for i in range(12)
        ]
    )
    m_const = compute_account_baselines(
        const_df,
        {"step": 350, "type": "TRANSFER", "amount": 500.0, "nameOrig": "O_CONST", "nameDest": "D_RENT"},
    )
    assert m_const.history_status == HistoryTier.ESTABLISHED
    assert m_const.low_history_confidence is False
    assert m_const.robust_z_score == pytest.approx(0.0)
    assert m_const.max_amount_ratio == pytest.approx(1.0)
    assert m_const.normality_band == NormalityBand.HIGHLY_NORMAL

    # 2. SPARSE (3 <= N_past < 10)
    sparse_df = pd.DataFrame(
        [
            {"step": 10, "type": "TRANSFER", "amount": 200.0, "nameOrig": "O_SPARSE", "nameDest": "D1"},
            {"step": 50, "type": "TRANSFER", "amount": 250.0, "nameOrig": "O_SPARSE", "nameDest": "D1"},
            {"step": 90, "type": "TRANSFER", "amount": 300.0, "nameOrig": "O_SPARSE", "nameDest": "D1"},
        ]
    )
    m_sparse = compute_account_baselines(
        sparse_df,
        {"step": 130, "type": "TRANSFER", "amount": 260.0, "nameOrig": "O_SPARSE", "nameDest": "D1"},
    )
    assert m_sparse.history_status == HistoryTier.SPARSE
    assert m_sparse.low_history_confidence is True
    assert m_sparse.normality_band == NormalityBand.AMBIGUOUS

    # 3. COLD_START (empty history + peer_df fallback -> safe, no ZeroDivisionError)
    peer_df = pd.DataFrame(
        [
            {"step": 10, "type": "TRANSFER", "amount": 100.0, "nameOrig": "P1", "nameDest": "D1"},
            {"step": 20, "type": "TRANSFER", "amount": 500.0, "nameOrig": "P2", "nameDest": "D2"},
            {"step": 30, "type": "TRANSFER", "amount": 1_000.0, "nameOrig": "P3", "nameDest": "D3"},
            {"step": 40, "type": "TRANSFER", "amount": 5_000.0, "nameOrig": "P4", "nameDest": "D4"},
        ]
    )
    m_cold = compute_account_baselines(
        pd.DataFrame(columns=["step", "type", "amount", "nameOrig", "nameDest"]),
        {"step": 100, "type": "TRANSFER", "amount": 750.0, "nameOrig": "O_NEW", "nameDest": "D_X"},
        peer_df=peer_df,
    )
    assert m_cold.history_status == HistoryTier.COLD_START
    assert m_cold.low_history_confidence is True
    assert m_cold.past_tx_count == 0
    assert m_cold.robust_z_score is None
    assert m_cold.max_amount_ratio is None
    assert m_cold.amount_percentile == pytest.approx(50.0)
    assert m_cold.normality_band == NormalityBand.AMBIGUOUS


def test_code_half_veto_rules_highly_normal_highly_abnormal_and_ambiguous() -> None:
    # 1. HIGHLY_NORMAL overrides 'approve' -> 'reject' (and -> 'request_more_evidence' in round 1 with chain flags)
    normal_metrics = CodeHalfMetrics(
        history_status=HistoryTier.ESTABLISHED,
        past_tx_count=15,
        history_span_steps=300,
        low_history_confidence=False,
        amount_percentile=60.0,
        max_amount_ratio=0.9,
        robust_z_score=0.5,
        velocity_1h_ratio=1.0,
        velocity_24h_ratio=1.0,
        inter_arrival_percentile=50.0,
        is_known_recipient=True,
        past_tx_to_recipient=5,
        recipient_volume_share=0.5,
        circular_hour_diff=1.0,
        periodic_recurrence_flag=True,
        normality_band=NormalityBand.HIGHLY_NORMAL,
        summary_text="normal",
    )
    assert classify_normality_band(normal_metrics) == NormalityBand.HIGHLY_NORMAL

    v_norm_rej, trig_1, _ = apply_code_veto(normal_metrics, "approve", review_round=1, has_external_chain_flags=False)
    assert v_norm_rej == "reject"
    assert trig_1 is True

    v_norm_chain, trig_2, _ = apply_code_veto(normal_metrics, "approve", review_round=1, has_external_chain_flags=True)
    assert v_norm_chain == "request_more_evidence"
    assert trig_2 is True

    # 2. HIGHLY_ABNORMAL overrides 'reject' -> 'request_more_evidence' (round 1) and -> 'approve' (round 2)
    abnormal_metrics = CodeHalfMetrics(
        history_status=HistoryTier.ESTABLISHED,
        past_tx_count=15,
        history_span_steps=300,
        low_history_confidence=False,
        amount_percentile=100.0,
        max_amount_ratio=6.5,
        robust_z_score=9.2,
        velocity_1h_ratio=4.0,
        velocity_24h_ratio=3.5,
        inter_arrival_percentile=1.0,
        is_known_recipient=False,
        past_tx_to_recipient=0,
        recipient_volume_share=0.0,
        circular_hour_diff=11.0,
        periodic_recurrence_flag=False,
        normality_band=NormalityBand.HIGHLY_ABNORMAL,
        summary_text="abnormal",
    )
    assert classify_normality_band(abnormal_metrics) == NormalityBand.HIGHLY_ABNORMAL

    v_abn_r1, trig_3, _ = apply_code_veto(abnormal_metrics, "reject", review_round=1)
    assert v_abn_r1 == "request_more_evidence"
    assert trig_3 is True

    v_abn_r2, trig_4, _ = apply_code_veto(abnormal_metrics, "reject", review_round=2)
    assert v_abn_r2 == "approve"
    assert trig_4 is True

    # 3. COLD_START / AMBIGUOUS allows LLM verdict through without hard veto
    cold_metrics = CodeHalfMetrics(
        history_status=HistoryTier.COLD_START,
        past_tx_count=0,
        history_span_steps=0,
        low_history_confidence=True,
        amount_percentile=50.0,
        max_amount_ratio=None,
        robust_z_score=None,
        velocity_1h_ratio=None,
        velocity_24h_ratio=None,
        inter_arrival_percentile=None,
        is_known_recipient=False,
        past_tx_to_recipient=0,
        recipient_volume_share=0.0,
        circular_hour_diff=None,
        periodic_recurrence_flag=False,
        normality_band=NormalityBand.AMBIGUOUS,
        summary_text="cold",
    )
    assert classify_normality_band(cold_metrics) == NormalityBand.AMBIGUOUS
    v_cold_app, trig_5, _ = apply_code_veto(cold_metrics, "approve", review_round=1)
    v_cold_rej, trig_6, _ = apply_code_veto(cold_metrics, "reject", review_round=1)
    assert v_cold_app == "approve" and trig_5 is False
    assert v_cold_rej == "reject" and trig_6 is False


def test_review_case_memory_firewall_retry_and_2_round_cap(tmp_path: Path) -> None:
    history_df = _make_established_history()
    # Ambiguous transaction (new recipient, moderate amount)
    ambiguous_case = {
        "case_id": "CASE_AMB_01",
        "review_round": 1,
        "transaction_ids": ["TXN_AMB_01"],
        "account_ids": {"nameOrig": "O_EST", "nameDest": "D_NEW_RECIPIENT"},
        "step": 500,
        "type": "TRANSFER",
        "amount": 1_200.0,
        "detector_scores": {"combined_anomaly_score": 0.72, "pattern_score": 0.30, "pattern_types": ""},
        "claim": "suspicious_requires_review",
        "evidence_for": ["New recipient"],
        "evidence_against": [],
        "confidence": 0.60,
    }
    sample_hints = [{"case_id": "MEM_FP_99", "final_label": 0, "note": "SECRET_MEMORY_TOKEN"}]

    # 1. Setup C MUST hide memory_hints from the LLM prompt (memory-blind)
    llm_c = _ScriptedReviewerLLM(
        responses=[
            json.dumps(
                {
                    "verdict": "approve",
                    "reviewer_confidence": 0.7,
                    "disproving_rationale": "Cannot disprove.",
                    "evidence_requests": [],
                }
            )
        ],
        log_path=tmp_path / "c.jsonl",
    )
    out_c = review_case(
        case=ambiguous_case,
        history_df=history_df,
        setup="C",
        review_round=1,
        memory_hints=sample_hints,
        llm_client=llm_c,
    )
    assert isinstance(out_c, ReviewerVerdictOutput)
    assert "SECRET_MEMORY_TOKEN" not in llm_c.captured_prompts[0]
    assert out_c.final_verdict == "approve"
    # Opinion-vs-truth firewall: no ground-truth label fields in serialized output
    serialized = out_c.to_dict()
    assert "final_label" not in serialized
    assert "label_source" not in serialized
    assert "verifier" not in serialized

    # 2. Setup D passes memory_hints to LLM Half (and validates retry on malformed JSON)
    llm_d = _ScriptedReviewerLLM(
        responses=[
            "not valid json!",
            json.dumps(
                {
                    "verdict": "reject",
                    "reviewer_confidence": 0.82,
                    "disproving_rationale": "Consistent with verified false-alarm hint.",
                    "evidence_requests": [],
                }
            ),
        ],
        log_path=tmp_path / "d.jsonl",
    )
    out_d = review_case(
        case=ambiguous_case,
        history_df=history_df,
        setup="D",
        review_round=1,
        memory_hints=sample_hints,
        llm_client=llm_d,
    )
    assert len(llm_d.captured_prompts) == 2  # retried once after invalid JSON
    assert "SECRET_MEMORY_TOKEN" in llm_d.captured_prompts[0]
    assert out_d.final_verdict == "reject"

    # 3. Setup D memory hint CANNOT override Code Half HIGHLY_ABNORMAL veto in round 2
    extreme_case = dict(ambiguous_case)
    extreme_case["case_id"] = "CASE_EXTREME"
    extreme_case["amount"] = 50_000.0  # > 3x max prior (1100) and robust_z >> 3.5, unknown recipient
    llm_veto = _ScriptedReviewerLLM(
        responses=[
            json.dumps(
                {
                    "verdict": "reject",
                    "reviewer_confidence": 0.85,
                    "disproving_rationale": "Attempting to reject based on memory hint.",
                    "evidence_requests": [],
                }
            )
        ],
        log_path=tmp_path / "veto.jsonl",
    )
    out_veto = review_case(
        case=extreme_case,
        history_df=history_df,
        setup="D",
        review_round=2,
        memory_hints=sample_hints,
        llm_client=llm_veto,
    )
    assert out_veto.code_half.normality_band == NormalityBand.HIGHLY_ABNORMAL
    assert out_veto.llm_raw_verdict == "reject"
    assert out_veto.final_verdict == "approve"
    assert out_veto.code_veto_triggered is True

    # 4. 2-round evidence cap converts request_more_evidence at round >= max_evidence_rounds -> approve
    llm_req = _ScriptedReviewerLLM(
        responses=[
            json.dumps(
                {
                    "verdict": "request_more_evidence",
                    "reviewer_confidence": 0.5,
                    "disproving_rationale": "Still unsure.",
                    "evidence_requests": [
                        {
                            "request_type": "CHECK_PERIODIC_HISTORY",
                            "target_account": "O_EST",
                            "target_recipient": None,
                            "window_steps": 168,
                            "reason": "Need more history.",
                        }
                    ],
                }
            )
        ],
        log_path=tmp_path / "cap.jsonl",
    )
    out_cap = review_case(
        case=ambiguous_case,
        history_df=history_df,
        setup="C",
        review_round=2,
        max_evidence_rounds=2,
        llm_client=llm_req,
    )
    assert out_cap.llm_raw_verdict == "request_more_evidence"
    assert out_cap.final_verdict == "approve"
    assert out_cap.evidence_requests == []
    assert "escalat" in out_cap.disproving_rationale.lower()
