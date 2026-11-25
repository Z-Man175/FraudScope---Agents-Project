"""LLM Half of the False-Positive Reviewer with schema validation, retries, and safe fallback.

WHAT IT DOES
    Sends canonicalized Code Half metrics + bounded InvestigationCase summaries to the LLM
    so it can attempt to disprove false-positive alerts (`approve`, `reject`, or
    `request_more_evidence`). Validates the JSON response, retries on malformed output,
    and falls back deterministically when running offline or after exhausting retries.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.orchestrator.llm_client import LLMClient, LLMConfig
from src.reviewer.prompts import PROMPT_VERSION, REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from src.reviewer.schemas import (
    CodeHalfMetrics,
    EvidenceRequest,
    EvidenceRequestType,
    NormalityBand,
    ReviewerVerdictType,
)

MODEL_VERSION = "local-deterministic-v1"
REVIEWER_SEED = 42

VALID_VERDICTS = {v.value for v in ReviewerVerdictType}
VALID_REQUEST_TYPES = {r.value for r in EvidenceRequestType}


def _strip_fences_and_parse(raw_text: str) -> dict[str, object]:
    """Parse JSON object from LLM text, tolerating markdown fences or brief preamble."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract outermost `{ ... }` block if the LLM added surrounding text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Reviewer LLM response must be a JSON object.")
    return parsed


def _parse_and_validate_llm_output(
    parsed: Mapping[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    """Validate Reviewer LLM output fields and convert `evidence_requests` into dataclasses."""
    errors: list[str] = []

    verdict = parsed.get("verdict")
    verdict_norm = str(verdict).strip().lower() if isinstance(verdict, str) else ""
    if verdict_norm not in VALID_VERDICTS:
        errors.append(f"'verdict' must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}.")

    conf = parsed.get("reviewer_confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errors.append("'reviewer_confidence' must be a float in [0.0, 1.0].")
    elif not (0.0 <= float(conf) <= 1.0):
        errors.append(f"'reviewer_confidence' out of bounds [0.0, 1.0]: {conf}.")

    rationale = parsed.get("disproving_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("'disproving_rationale' must be a non-empty string.")

    raw_reqs = parsed.get("evidence_requests", [])
    parsed_reqs: list[EvidenceRequest] = []
    if not isinstance(raw_reqs, list):
        errors.append("'evidence_requests' must be a list.")
    else:
        for idx, item in enumerate(raw_reqs):
            if not isinstance(item, Mapping):
                errors.append(f"evidence_requests[{idx}] must be an object.")
                continue
            rt_raw = str(item.get("request_type", "")).strip().upper()
            if rt_raw not in VALID_REQUEST_TYPES:
                errors.append(
                    f"evidence_requests[{idx}].request_type '{rt_raw}' not in {sorted(VALID_REQUEST_TYPES)}."
                )
                continue
            win = item.get("window_steps")
            win_val = int(win) if isinstance(win, (int, float)) and not isinstance(win, bool) else None
            parsed_reqs.append(
                EvidenceRequest(
                    request_type=EvidenceRequestType(rt_raw),
                    target_account=str(item["target_account"]) if item.get("target_account") else None,
                    target_recipient=str(item["target_recipient"]) if item.get("target_recipient") else None,
                    window_steps=win_val,
                    reason=str(item.get("reason", "")),
                )
            )

    if errors:
        return None, errors

    return {
        "verdict": verdict_norm,
        "reviewer_confidence": round(float(conf), 4),
        "disproving_rationale": str(rationale).strip(),
        "evidence_requests": parsed_reqs,
    }, []


def _deterministic_reviewer_decision(
    case_dict: Mapping[str, object],
    code_metrics: CodeHalfMetrics,
    review_round: int,
    memory_hints: list[dict[str, object]] | None,
    has_external_chain_flags: bool,
) -> dict[str, object]:
    """Deterministic fallback decision grounded in Code Half metrics & bounded case summary."""
    acct_ids = case_dict.get("account_ids", {})
    orig = str(acct_ids.get("nameOrig", "")) if isinstance(acct_ids, Mapping) else ""
    dest = str(acct_ids.get("nameDest", "")) if isinstance(acct_ids, Mapping) else ""

    inv_conf = float(case_dict.get("confidence", 0.5) or 0.5)
    band = code_metrics.normality_band

    if band == NormalityBand.HIGHLY_NORMAL:
        # Even for HIGHLY_NORMAL accounts, verify downstream hops in round 1 if chain flags exist
        if review_round == 1 and has_external_chain_flags:
            return {
                "verdict": ReviewerVerdictType.REQUEST_MORE_EVIDENCE.value,
                "reviewer_confidence": 0.65,
                "disproving_rationale": (
                    "Account history is HIGHLY_NORMAL, but external chain flags warrant "
                    "tracing downstream hops before rejecting."
                ),
                "evidence_requests": [
                    EvidenceRequest(
                        request_type=EvidenceRequestType.TRACE_DOWNSTREAM_HOPS,
                        target_account=orig or None,
                        target_recipient=dest or None,
                        window_steps=24,
                        reason="Verify whether funds continue hopping downstream.",
                    )
                ],
            }
        return {
            "verdict": ReviewerVerdictType.REJECT.value,
            "reviewer_confidence": 0.9,
            "disproving_rationale": (
                f"Disproved as likely false positive: {code_metrics.summary_text}"
            ),
            "evidence_requests": [],
        }

    if band == NormalityBand.HIGHLY_ABNORMAL:
        return {
            "verdict": ReviewerVerdictType.APPROVE.value,
            "reviewer_confidence": round(min(max(inv_conf, 0.85), 0.99), 4),
            "disproving_rationale": (
                f"Cannot disprove fraud claim: {code_metrics.summary_text}"
            ),
            "evidence_requests": [],
        }

    # AMBIGUOUS / SPARSE / COLD_START: weigh Setup D memory hints and investigator confidence
    if memory_hints:
        fp_hints = [
            h
            for h in memory_hints
            if isinstance(h, Mapping)
            and str(h.get("final_label", "")).lower() in {"0", "false_positive", "legit"}
        ]
        if fp_hints and inv_conf < 0.85:
            return {
                "verdict": ReviewerVerdictType.REJECT.value,
                "reviewer_confidence": 0.78,
                "disproving_rationale": (
                    f"Ambiguous baseline supported by {len(fp_hints)} similar confirmed "
                    "false-alarm memory hint(s); rejecting alert."
                ),
                "evidence_requests": [],
            }

    if inv_conf >= 0.75:
        return {
            "verdict": ReviewerVerdictType.APPROVE.value,
            "reviewer_confidence": round(min(inv_conf, 0.95), 4),
            "disproving_rationale": (
                f"Ambiguous/limited history ({code_metrics.history_status.value}) cannot disprove "
                f"high-confidence fraud indicators (confidence={inv_conf:.2f})."
            ),
            "evidence_requests": [],
        }

    if inv_conf <= 0.35 and (code_metrics.is_known_recipient or code_metrics.periodic_recurrence_flag):
        return {
            "verdict": ReviewerVerdictType.REJECT.value,
            "reviewer_confidence": 0.75,
            "disproving_rationale": (
                "Low investigator confidence combined with known recipient or periodic history "
                "suggests benign activity."
            ),
            "evidence_requests": [],
        }

    if review_round == 1:
        req_type = (
            EvidenceRequestType.CHECK_PERIODIC_HISTORY
            if not code_metrics.periodic_recurrence_flag
            else EvidenceRequestType.QUERY_RECIPIENT_FAN_IN
        )
        return {
            "verdict": ReviewerVerdictType.REQUEST_MORE_EVIDENCE.value,
            "reviewer_confidence": 0.55,
            "disproving_rationale": (
                f"Ambiguous baseline ({code_metrics.history_status.value}); requesting targeted "
                f"evidence ({req_type.value}) to resolve false-positive risk."
            ),
            "evidence_requests": [
                EvidenceRequest(
                    request_type=req_type,
                    target_account=orig or None,
                    target_recipient=dest or None,
                    window_steps=168 if req_type == EvidenceRequestType.CHECK_PERIODIC_HISTORY else 24,
                    reason="Resolve ambiguous account baseline before final verdict.",
                )
            ],
        }

    verdict = (
        ReviewerVerdictType.APPROVE.value
        if inv_conf >= 0.5
        else ReviewerVerdictType.REJECT.value
    )
    return {
        "verdict": verdict,
        "reviewer_confidence": round(min(max(inv_conf, 0.5), 0.85), 4),
        "disproving_rationale": (
            f"Final round decision on ambiguous baseline ({code_metrics.history_status.value}) "
            f"with investigator confidence={inv_conf:.2f}."
        ),
        "evidence_requests": [],
    }


def run_llm_reviewer(
    case_dict: Mapping[str, object],
    code_metrics: CodeHalfMetrics,
    review_round: int = 1,
    memory_hints: list[dict[str, object]] | None = None,
    has_external_chain_flags: bool = False,
    llm_client: LLMClient | Any | None = None,
    max_retries: int = 2,
) -> dict[str, object]:
    """Invoke the LLM Half with deterministic settings, schema validation, and retries."""
    client = llm_client or LLMClient(
        LLMConfig(
            model_version=MODEL_VERSION,
            prompt_version=PROMPT_VERSION,
            temperature=0.0,
            top_p=1.0,
            seed=REVIEWER_SEED,
        )
    )
    case_id = str(case_dict.get("case_id", "CASE_UNKNOWN"))
    base_prompt = build_reviewer_prompt(
        case_payload=case_dict,
        code_metrics=code_metrics,
        review_round=review_round,
        memory_hints=memory_hints,
    )
    schema_hint = {
        "schema_name": "ReviewerVerdictOutput",
        "required_keys": [
            "verdict",
            "reviewer_confidence",
            "disproving_rationale",
            "evidence_requests",
        ],
    }

    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        step_name = f"reviewer_round_{review_round}"
        if attempt > 0:
            step_name = f"reviewer_round_{review_round}_retry_{attempt}"
            prompt = (
                f"{base_prompt}\n\nPrevious output failed validation with errors: {last_errors}. "
                "Return ONLY valid JSON matching the Reviewer schema."
            )

        resp = client.generate(
            prompt=prompt,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            case_id=case_id,
            step_name=step_name,
        )

        try:
            parsed = _strip_fences_and_parse(resp.content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_errors = [f"JSON parse error: {exc}"]
            continue

        # When running on the offline generic LLMClient stub (which emits `status: "ok"`),
        # route directly to `_deterministic_reviewer_decision` without wasting retry attempts.
        if (
            getattr(resp, "mode", None) == "deterministic_stub"
            and "verdict" not in parsed
            and parsed.get("status") == "ok"
        ):
            return _deterministic_reviewer_decision(
                case_dict=case_dict,
                code_metrics=code_metrics,
                review_round=review_round,
                memory_hints=memory_hints,
                has_external_chain_flags=has_external_chain_flags,
            )

        validated, val_errors = _parse_and_validate_llm_output(parsed)
        if validated is not None:
            return validated
        last_errors = val_errors

    # Safe fallback after exhausting retries
    fallback = _deterministic_reviewer_decision(
        case_dict=case_dict,
        code_metrics=code_metrics,
        review_round=review_round,
        memory_hints=memory_hints,
        has_external_chain_flags=has_external_chain_flags,
    )
    fallback["disproving_rationale"] = (
        f"{fallback['disproving_rationale']} [Validation fallback after {max_retries} retries]"
    )
    return fallback