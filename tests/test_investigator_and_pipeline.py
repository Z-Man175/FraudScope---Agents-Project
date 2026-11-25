"""Unit and integration tests for LLMClient, Investigation Agent, and Orchestrator Pipeline."""

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
from src.agents.investigator_agent import (
    InvestigationCase,
    fulfill_evidence_request,
    investigate_candidate,
    query_account_history,
    query_linked_accounts,
    query_recipient_history,
    validate_investigation_case,
)
from src.orchestrator.pipeline import SetupMode, run_pipeline


def _sample_history_df() -> pd.DataFrame:
    """Create a multi-step transaction table with past, current, and future rows."""
    return pd.DataFrame(
        [
            # Periodic prior history for O_PERIODIC (every 168 steps: 100, 268, 436)
            {"txn_id": "T001", "step": 100, "type": "TRANSFER", "amount": 20_000.0, "nameOrig": "O_PERIODIC", "nameDest": "D_EMP_1", "isFraud": 0},
            {"txn_id": "T002", "step": 268, "type": "TRANSFER", "amount": 20_000.0, "nameOrig": "O_PERIODIC", "nameDest": "D_EMP_1", "isFraud": 0},
            {"txn_id": "T003", "step": 436, "type": "TRANSFER", "amount": 20_000.0, "nameOrig": "O_PERIODIC", "nameDest": "D_EMP_1", "isFraud": 0},
            # Past fan-in to D_MULE (steps 480..495)
            {"txn_id": "T004", "step": 480, "type": "TRANSFER", "amount": 3_000.0, "nameOrig": "O_S1", "nameDest": "D_MULE", "isFraud": 1},
            {"txn_id": "T005", "step": 485, "type": "TRANSFER", "amount": 3_200.0, "nameOrig": "O_S2", "nameDest": "D_MULE", "isFraud": 1},
            {"txn_id": "T006", "step": 490, "type": "CASH_OUT", "amount": 2_900.0, "nameOrig": "O_S3", "nameDest": "D_MULE", "isFraud": 1},
            # Past 2-hop downstream chain from O_CHAIN: O_CHAIN -> H1 -> H2 before step 500
            {"txn_id": "T007", "step": 492, "type": "TRANSFER", "amount": 70_000.0, "nameOrig": "O_CHAIN", "nameDest": "H1", "isFraud": 1},
            {"txn_id": "T008", "step": 495, "type": "TRANSFER", "amount": 68_000.0, "nameOrig": "H1", "nameDest": "H2", "isFraud": 1},
            # Candidate step == 500 (must NOT appear in step < 500 tool queries)
            {"txn_id": "T009", "step": 500, "type": "TRANSFER", "amount": 95_000.0, "nameOrig": "O_CHAIN", "nameDest": "D_MULE", "isFraud": 1},
            # Future leakage trap rows at step > 500 (must NEVER appear in step < 500 queries)
            {"txn_id": "T010", "step": 505, "type": "TRANSFER", "amount": 500_000.0, "nameOrig": "O_CHAIN", "nameDest": "H_FUTURE", "isFraud": 1},
            {"txn_id": "T011", "step": 510, "type": "TRANSFER", "amount": 4_000.0, "nameOrig": "O_FUTURE_SENDER", "nameDest": "D_MULE", "isFraud": 1},
        ]
    )


def test_llm_client_deterministic_tracking_and_jsonl_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "llm_calls.jsonl"
    cfg = LLMConfig(
        model_version="local-deterministic-v1",
        prompt_version="v1",
        temperature=0.0,
        top_p=1.0,
        seed=42,
        cost_per_1k_input_tokens_usd=0.001,
        cost_per_1k_output_tokens_usd=0.002,
        log_path=log_file,
    )
    client = LLMClient(cfg)

    resp_1 = client.generate("Summarize evidence A", system_prompt="System", case_id="CASE_1", step_name="test_call")
    resp_2 = client.generate("Summarize evidence A", system_prompt="System", case_id="CASE_1", step_name="test_call")

    assert resp_1.mode == "deterministic_stub"
    assert resp_1.content == resp_2.content
    assert resp_1.input_hash == resp_2.input_hash
    assert resp_1.output_hash == resp_2.output_hash
    assert resp_1.seed == 42
    assert client.call_count == 2
    assert client.total_estimated_cost_usd > 0.0

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    logged = json.loads(lines[0])
    assert logged["model_version"] == "local-deterministic-v1"
    assert logged["prompt_version"] == "v1"
    assert logged["temperature"] == 0.0
    assert logged["seed"] == 42
    assert logged["case_id"] == "CASE_1"
    assert logged["mode"] == "deterministic_stub"


def test_evidence_tools_enforce_strict_past_and_fulfill_requests() -> None:
    df = _sample_history_df()

    # Account history strictly step < 500 must exclude step 500 and step 505
    acct_hist = query_account_history(df, "O_CHAIN", max_step=500)
    assert acct_hist["prior_txn_count"] == 1
    assert acct_hist["last_step"] == 492

    # Recipient fan-in strictly step < 500 must exclude T009 (step 500) and T011 (step 510)
    recip_hist = query_recipient_history(df, "D_MULE", max_step=500)
    assert recip_hist["incoming_count"] == 3
    assert recip_hist["unique_senders"] == 3
    assert "O_FUTURE_SENDER" not in recip_hist["sender_sample"]

    # Linked accounts strictly step < 500 must see H1 -> H2 but NOT H_FUTURE
    linked = query_linked_accounts(df, "O_CHAIN", max_step=500, max_hops=2)
    assert linked["outgoing_hops"]["hop_1"] == ["H1"]
    assert linked["outgoing_hops"]["hop_2"] == ["H2"]
    assert "H_FUTURE" not in str(linked)

    # Fulfill all four evidence request types
    periodic_ev = fulfill_evidence_request(
        df,
        "CHECK_PERIODIC_HISTORY",
        {"account_id": "O_PERIODIC"},
        max_step=604,
    )
    assert periodic_ev["is_periodic"] is True
    assert periodic_ev["dominant_interval_steps"] == pytest.approx(168.0)

    fan_in_ev = fulfill_evidence_request(
        df,
        "QUERY_RECIPIENT_FAN_IN",
        {"dest_account": "D_MULE"},
        max_step=500,
    )
    assert fan_in_ev["unique_senders"] == 3

    hops_ev = fulfill_evidence_request(
        df,
        "TRACE_DOWNSTREAM_HOPS",
        {"account_id": "O_CHAIN", "max_hops": 2},
        max_step=500,
    )
    assert hops_ev["outgoing_hops"]["hop_2"] == ["H2"]

    mem_ev = fulfill_evidence_request(
        df,
        "FETCH_SIMILAR_FALSE_ALARMS",
        {"account_id": "O_PERIODIC"},
        max_step=604,
    )
    assert mem_ev["memory_available"] is False
    assert mem_ev["matches"] == []


class _FlakyLLMClient(LLMClient):
    """Returns invalid JSON on first call, then valid structured JSON on retry."""

    def __init__(self, config: LLMConfig, fail_times: int = 1) -> None:
        super().__init__(config)
        self._fail_times = fail_times
        self._seen = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: dict | None = None,
        case_id: str | None = None,
        step_name: str = "unknown",
    ) -> LLMResponse:
        self._seen += 1
        if self._seen <= self._fail_times:
            # Log a malformed JSON payload via base stub path by overriding content
            resp = super().generate(prompt, system_prompt, schema_hint, case_id, step_name)
            resp.content = "{not valid json"
            return resp
        return super().generate(prompt, system_prompt, schema_hint, case_id, step_name)


def test_investigation_case_validation_and_retry_behavior(tmp_path: Path) -> None:
    df = _sample_history_df()
    cand = {
        "txn_id": "T009",
        "step": 500,
        "type": "TRANSFER",
        "amount": 95_000.0,
        "nameOrig": "O_CHAIN",
        "nameDest": "D_MULE",
        "combined_anomaly_score": 0.92,
        "pattern_score": 0.85,
        "pattern_types": "account_chain",
    }

    # 1. Flaky LLM fails once, succeeds on retry 1
    flaky_client = _FlakyLLMClient(LLMConfig(log_path=tmp_path / "retry.jsonl"), fail_times=1)
    case = investigate_candidate(cand, df, llm_client=flaky_client, max_retries=2)
    assert isinstance(case, InvestigationCase)
    assert flaky_client.call_count == 2
    assert validate_investigation_case(case.to_dict()) == []
    assert case.review_round == 1
    assert 0.0 <= case.confidence <= 1.0

    # 2. Always-broken LLM exhausts retries and returns safe schema-valid fallback
    broken_client = _FlakyLLMClient(LLMConfig(log_path=tmp_path / "fallback.jsonl"), fail_times=10)
    fallback_case = investigate_candidate(cand, df, llm_client=broken_client, max_retries=2)
    assert broken_client.call_count == 3  # initial + 2 retries
    assert validate_investigation_case(fallback_case.to_dict()) == []
    assert fallback_case.tool_summaries["validation_fallback"]["used"] is True


def test_pipeline_setups_b_c_d_escalation_and_label_separation(tmp_path: Path) -> None:
    history_df = _sample_history_df()
    candidates_df = pd.DataFrame(
        [
            {
                "txn_id": "T009",
                "step": 500,
                "type": "TRANSFER",
                "amount": 95_000.0,
                "nameOrig": "O_CHAIN",
                "nameDest": "D_MULE",
                "isFraud": 1,
                "combined_anomaly_score": 0.92,
                "pattern_score": 0.85,
                "pattern_types": "account_chain",
                "ranking_score": 0.89,
                "rank": 1,
            },
            {
                "txn_id": "T_PER_NEW",
                "step": 604,
                "type": "TRANSFER",
                "amount": 20_000.0,
                "nameOrig": "O_PERIODIC",
                "nameDest": "D_EMP_1",
                "isFraud": 0,
                "combined_anomaly_score": 0.72,
                "pattern_score": 0.40,
                "pattern_types": "burst",
                "ranking_score": 0.62,
                "rank": 2,
            },
        ]
    )

    # --- Setup B: Agents without reviewer ---
    res_b = run_pipeline(
        candidates_df=candidates_df,
        history_df=history_df,
        setup=SetupMode.B,
        llm_client=LLMClient(LLMConfig(log_path=tmp_path / "b_llm.jsonl")),
        out_dir=tmp_path / "out_b",
    )
    assert len(res_b.records) == 2
    assert all(r["reviewer_verdict"] is None for r in res_b.records)
    assert all(r["status"] == "INVESTIGATED" for r in res_b.records)
    assert all(r["final_label"] is None for r in res_b.records)
    assert (tmp_path / "out_b" / "setup_b_cases.json").exists()

    # --- Setup C: Reviewer approves T009, rejects T_PER_NEW after 1 evidence round; memory is blind ---
    def mock_reviewer_c(case: InvestigationCase, past_hist: pd.DataFrame) -> dict[str, object]:
        # Verify reviewer receives strictly past history (step < candidate step)
        cand_step = 500 if case.transaction_ids == ["T009"] else 604
        assert (past_hist["step"] < cand_step).all()
        assert case.memory_hints == []  # Setup C must stay memory-blind

        if case.transaction_ids == ["T009"]:
            return {"verdict": "approve", "reason": "Strong chain + fan-in evidence."}
        if case.review_round == 1:
            return {
                "verdict": "request_more_evidence",
                "reason": "Check whether origin has periodic transfer cadence.",
                "evidence_requests": [
                    {"request_type": "CHECK_PERIODIC_HISTORY", "payload": {"account_id": "O_PERIODIC"}}
                ],
            }
        assert case.tool_summaries["CHECK_PERIODIC_HISTORY"]["is_periodic"] is True
        return {"verdict": "reject", "reason": "Confirmed periodic benign cadence."}

    res_c = run_pipeline(
        candidates_df=candidates_df,
        history_df=history_df,
        setup="C",
        max_evidence_rounds=2,
        llm_client=LLMClient(LLMConfig(log_path=tmp_path / "c_llm.jsonl")),
        reviewer_fn=mock_reviewer_c,
        # Even if a retriever is passed, Setup C must NOT use it
        memory_retriever=lambda q: [{"case_id": "SHOULD_NOT_APPEAR", "final_label": 0}],
        out_dir=tmp_path / "out_c",
    )
    rec_fraud, rec_legit = res_c.records
    assert rec_fraud["reviewer_verdict"] == "approve"
    assert rec_fraud["status"] == "APPROVED_FOR_ALERT"
    # Reviewer approval must NEVER become final_label on its own
    assert rec_fraud["final_label"] is None
    assert rec_fraud["label_source"] is None
    assert rec_fraud["verifier"] is None

    assert rec_legit["reviewer_verdict"] == "reject"
    assert rec_legit["status"] == "REJECTED_FALSE_POSITIVE"
    assert rec_legit["evidence_rounds_used"] == 1
    assert rec_legit["final_label"] is None

    # --- Escalation after 2 follow-up evidence rounds ---
    def always_request_reviewer(case: InvestigationCase, past_hist: pd.DataFrame) -> dict[str, object]:
        return {
            "verdict": "request_more_evidence",
            "reason": "Need more context.",
            "evidence_requests": [{"request_type": "QUERY_RECIPIENT_FAN_IN", "payload": {}}],
        }

    res_esc = run_pipeline(
        candidates_df=candidates_df.head(1),
        history_df=history_df,
        setup="C",
        max_evidence_rounds=2,
        llm_client=LLMClient(LLMConfig(log_path=tmp_path / "esc_llm.jsonl")),
        reviewer_fn=always_request_reviewer,
        out_dir=tmp_path / "out_esc",
    )
    esc_rec = res_esc.records[0]
    assert esc_rec["evidence_rounds_used"] == 2
    assert esc_rec["status"] == "ESCALATED"
    assert esc_rec["reviewer_verdict"] == "escalate"
    assert esc_rec["final_label"] is None

    # --- Setup D: Uses memory retriever, integrity check, and only writes confirmed labels ---
    written_to_memory: list[dict[str, object]] = []

    def mock_reviewer_d(case: InvestigationCase, past_hist: pd.DataFrame) -> dict[str, object]:
        assert len(case.memory_hints) == 1
        return {"verdict": "approve", "reason": "Matches prior verified fraud pattern."}

    res_d = run_pipeline(
        candidates_df=candidates_df,
        history_df=history_df,
        setup="D",
        llm_client=LLMClient(LLMConfig(log_path=tmp_path / "d_llm.jsonl")),
        reviewer_fn=mock_reviewer_d,
        memory_retriever=lambda q: [{"case_id": "MEM_001", "final_label": 1, "pattern": "account_chain"}],
        memory_writer=lambda rec: written_to_memory.append(rec),
        integrity_checker=lambda: True,
        # Only T009 has an external ground-truth confirmation; T_PER_NEW remains unconfirmed
        confirmation_labels={"T009": {"final_label": 1, "label_source": "ground_truth_oracle", "verifier": "analyst_1"}},
        out_dir=tmp_path / "out_d",
    )
    assert res_d.summary["integrity_check_passed"] is True
    assert res_d.summary["confirmed_count"] == 1
    assert res_d.records[0]["final_label"] == 1
    assert res_d.records[0]["label_source"] == "ground_truth_oracle"
    assert res_d.records[0]["verifier"] == "analyst_1"
    assert res_d.records[1]["final_label"] is None
    # Only the confirmed case was written to verified memory
    assert len(written_to_memory) == 1
    assert written_to_memory[0]["txn_id"] == "T009"
