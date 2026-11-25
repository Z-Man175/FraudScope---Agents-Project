"""Investigation Agent with classical leak-free evidence tools and validated Case JSON.

WHAT IT DOES
    1. Runs deterministic, strictly past-only (`step < max_step`) pandas/numpy tools to
       summarize account history, recipient fan-in, and 2-hop fund neighborhoods.
    2. Sends ONLY bounded summaries + detector/pattern metadata (never raw tables or
       `isFraud` labels) to the LLM to synthesize a structured `InvestigationCase`.
    3. Validates the LLM output against the `InvestigationCase` schema, retries on
       malformed JSON, and falls back deterministically if retries are exhausted.
    4. Fulfills follow-up evidence requests from the Reviewer (`update_case_with_evidence`)
       for bounded multi-round investigation (Setups C and D).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable, Mapping
import numpy as np
import pandas as pd

from src.orchestrator.llm_client import LLMClient, LLMConfig

# Whitelist of follow-up tool requests the Reviewer agent is allowed to trigger
SUPPORTED_EVIDENCE_REQUESTS = {
    "CHECK_PERIODIC_HISTORY",
    "QUERY_RECIPIENT_FAN_IN",
    "TRACE_DOWNSTREAM_HOPS",
    "FETCH_SIMILAR_FALSE_ALARMS",
}

INVESTIGATOR_SYSTEM_PROMPT = (
    "You are the FraudScope Investigation Agent. You only receive structured tool summaries "
    "computed over strictly past transactions (step < candidate_step). Return ONLY a valid "
    "JSON object with keys: claim (str), evidence_for (list[str]), "
    "evidence_against (list[str]), confidence (float in [0.0, 1.0])."
)


@dataclass
class InvestigationCase:
    """Structured case dossier passed between the Investigator, Reviewer, and Orchestrator."""

    case_id: str
    review_round: int
    transaction_ids: list[str]
    account_ids: dict[str, str]
    detector_scores: dict[str, object]
    tool_summaries: dict[str, object]
    claim: str
    evidence_for: list[str]
    evidence_against: list[str]
    confidence: float
    memory_hints: list[dict[str, object]] = field(default_factory=list)
    prompt_version: str = "v1"
    model_version: str = "local-deterministic-v1"
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "InvestigationCase":
        errors = validate_investigation_case(data)
        if errors:
            raise ValueError(f"Invalid InvestigationCase payload: {'; '.join(errors)}")
        return cls(
            case_id=str(data["case_id"]),
            review_round=int(data["review_round"]),
            transaction_ids=[str(t) for t in data["transaction_ids"]],  # type: ignore[index]
            account_ids={
                "nameOrig": str(data["account_ids"]["nameOrig"]),  # type: ignore[index]
                "nameDest": str(data["account_ids"]["nameDest"]),  # type: ignore[index]
            },
            detector_scores=dict(data["detector_scores"]),  # type: ignore[arg-type]
            tool_summaries=dict(data["tool_summaries"]),  # type: ignore[arg-type]
            claim=str(data["claim"]),
            evidence_for=[str(x) for x in data["evidence_for"]],  # type: ignore[union-attr]
            evidence_against=[str(x) for x in data["evidence_against"]],  # type: ignore[union-attr]
            confidence=float(data["confidence"]),
            memory_hints=list(data.get("memory_hints") or []),  # type: ignore[arg-type]
            prompt_version=str(data["prompt_version"]),
            model_version=str(data["model_version"]),
            seed=int(data["seed"]),
        )


def validate_investigation_case(data: Mapping[str, object]) -> list[str]:
    """Validate structured `InvestigationCase` fields and types; return list of errors."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return ["Case payload must be a mapping/dict."]

    required_fields = [
        "case_id",
        "review_round",
        "transaction_ids",
        "account_ids",
        "detector_scores",
        "tool_summaries",
        "claim",
        "evidence_for",
        "evidence_against",
        "confidence",
        "prompt_version",
        "model_version",
        "seed",
    ]
    for key in required_fields:
        if key not in data:
            errors.append(f"Missing required field '{key}'.")
    if errors:
        return errors

    if not isinstance(data["case_id"], str) or not data["case_id"].strip():
        errors.append("'case_id' must be a non-empty string.")

    # Note: isinstance(True, int) is True in Python, so explicitly reject booleans
    review_round = data["review_round"]
    if isinstance(review_round, bool) or not isinstance(review_round, int) or review_round < 1:
        errors.append("'review_round' must be an integer >= 1.")

    txn_ids = data["transaction_ids"]
    if (
        not isinstance(txn_ids, list)
        or not txn_ids
        or not all(isinstance(t, str) and t.strip() for t in txn_ids)
    ):
        errors.append("'transaction_ids' must be a non-empty list of strings.")

    account_ids = data["account_ids"]
    if not isinstance(account_ids, Mapping):
        errors.append("'account_ids' must be a dict with 'nameOrig' and 'nameDest'.")
    else:
        for acct_key in ("nameOrig", "nameDest"):
            val = account_ids.get(acct_key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"'account_ids.{acct_key}' must be a non-empty string.")

    if not isinstance(data["detector_scores"], Mapping):
        errors.append("'detector_scores' must be a dict.")

    if not isinstance(data["tool_summaries"], Mapping):
        errors.append("'tool_summaries' must be a dict.")

    if not isinstance(data["claim"], str) or not data["claim"].strip():
        errors.append("'claim' must be a non-empty string.")

    for list_field in ("evidence_for", "evidence_against"):
        val = data[list_field]
        if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
            errors.append(f"'{list_field}' must be a list of strings.")

    conf = data["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        errors.append("'confidence' must be a float in [0.0, 1.0].")
    elif not (0.0 <= float(conf) <= 1.0):
        errors.append(f"'confidence' must be within [0.0, 1.0], got {conf}.")

    mem_hints = data.get("memory_hints")
    if mem_hints is not None and not isinstance(mem_hints, list):
        errors.append("'memory_hints' must be a list or None.")

    if not isinstance(data["prompt_version"], str) or not data["prompt_version"].strip():
        errors.append("'prompt_version' must be a non-empty string.")

    if not isinstance(data["model_version"], str) or not data["model_version"].strip():
        errors.append("'model_version' must be a non-empty string.")

    seed_val = data["seed"]
    if isinstance(seed_val, bool) or not isinstance(seed_val, int):
        errors.append("'seed' must be an integer.")

    return errors


# ---------------------------------------------------------------------------
# Classical Leak-Free Evidence Tools (strictly step < max_step)
# ---------------------------------------------------------------------------


def _strict_past(df: pd.DataFrame, max_step: int) -> pd.DataFrame:
    """Enforce zero future leakage: exclude the candidate's own step and all later steps."""
    if df.empty:
        return df
    return df[df["step"] < int(max_step)]


def _analyze_periodicity(steps: np.ndarray, amounts: np.ndarray) -> dict[str, object]:
    """Check whether sorted transaction steps exhibit consistent periodic cadence.

    Designed to help separate legitimate recurring look-alikes (e.g., weekly payroll or
    rent every 168 steps in `scenarios.py`) from one-off fraud bursts.
    """
    unique_steps = np.unique(steps)
    if len(unique_steps) < 2:
        return {
            "is_periodic": False,
            "dominant_interval_steps": None,
            "interval_cv": None,
            "amount_cv": None,
            "prior_cycle_count": int(len(unique_steps)),
        }

    # Collapse multi-transaction batches (e.g., 5 payroll transfers across 2 adjacent steps)
    # into a single cycle start timestamp before measuring inter-cycle intervals
    cycle_starts = [int(unique_steps[0])]
    for s in unique_steps[1:]:
        if int(s) - cycle_starts[-1] > 2:
            cycle_starts.append(int(s))

    if len(cycle_starts) < 2:
        return {
            "is_periodic": False,
            "dominant_interval_steps": None,
            "interval_cv": None,
            "amount_cv": None,
            "prior_cycle_count": int(len(cycle_starts)),
        }

    gaps = np.diff(np.asarray(cycle_starts, dtype=np.float64))
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    # Coefficient of variation (std / mean): low CV means highly regular timing
    interval_cv = float(std_gap / mean_gap) if mean_gap > 0.0 else None

    mean_amt = float(np.mean(amounts)) if len(amounts) > 0 else 0.0
    std_amt = float(np.std(amounts)) if len(amounts) > 0 else 0.0
    amount_cv = float(std_amt / mean_amt) if mean_amt > 0.0 else None

    # Regular cadence requires >= 2 prior cycles spaced >= 24 steps apart with <= 15% jitter
    is_regular_cadence = (
        len(cycle_starts) >= 2
        and mean_gap >= 24.0
        and (interval_cv is not None and interval_cv <= 0.15)
    )
    return {
        "is_periodic": bool(is_regular_cadence),
        "dominant_interval_steps": round(mean_gap, 2),
        "interval_cv": round(interval_cv, 4) if interval_cv is not None else None,
        "amount_cv": round(amount_cv, 4) if amount_cv is not None else None,
        "prior_cycle_count": int(len(cycle_starts)),
    }


def query_account_history(
    df: pd.DataFrame,
    account_id: str,
    max_step: int,
) -> dict[str, object]:
    """Summarize strictly past (`step < max_step`) transaction history for an account."""
    past = _strict_past(df, max_step)
    orig_mask = past["nameOrig"].astype(str) == str(account_id)
    dest_mask = past["nameDest"].astype(str) == str(account_id)
    acct_df = past[orig_mask | dest_mask]

    if acct_df.empty:
        return {
            "account_id": str(account_id),
            "max_step": int(max_step),
            "prior_txn_count": 0,
            "prior_outgoing_count": 0,
            "prior_incoming_count": 0,
            "outgoing_amount_mean": None,
            "outgoing_amount_std": None,
            "outgoing_amount_max": None,
            "unique_recipients": 0,
            "unique_senders": 0,
            "first_step": None,
            "last_step": None,
            "periodic_hint": False,
            "dominant_interval_steps": None,
            "type_counts": {},
            "recent_txns": [],
        }

    acct_df = acct_df.sort_values(["step"], kind="mergesort")
    out_df = past[orig_mask].sort_values(["step"], kind="mergesort")
    in_df = past[dest_mask].sort_values(["step"], kind="mergesort")

    out_amts = out_df["amount"].to_numpy(dtype=np.float64)
    # Analyze periodicity on outgoing history when available so incoming deposits do not
    # distort recurring payment intervals; fall back to combined history for pure recipients
    cadence_df = out_df if len(out_df) >= 2 else acct_df
    periodicity = _analyze_periodicity(
        steps=cadence_df["step"].to_numpy(dtype=np.int64),
        amounts=cadence_df["amount"].to_numpy(dtype=np.float64),
    )

    type_counts: dict[str, int] = {}
    if "type" in acct_df.columns:
        type_counts = {
            str(k): int(v) for k, v in acct_df["type"].value_counts().sort_index().items()
        }

    # Bound LLM context to the 5 most recent transactions before candidate step
    recent_rows: list[dict[str, object]] = []
    for row in acct_df.tail(5).itertuples(index=False):
        direction = "outgoing" if str(row.nameOrig) == str(account_id) else "incoming"
        counterparty = str(row.nameDest) if direction == "outgoing" else str(row.nameOrig)
        entry: dict[str, object] = {
            "step": int(row.step),
            "direction": direction,
            "counterparty": counterparty,
            "amount": round(float(row.amount), 2),
        }
        if hasattr(row, "txn_id"):
            entry["txn_id"] = str(row.txn_id)
        if hasattr(row, "type"):
            entry["type"] = str(row.type)
        recent_rows.append(entry)

    return {
        "account_id": str(account_id),
        "max_step": int(max_step),
        "prior_txn_count": int(len(acct_df)),
        "prior_outgoing_count": int(len(out_df)),
        "prior_incoming_count": int(len(in_df)),
        "outgoing_amount_mean": round(float(np.mean(out_amts)), 2) if len(out_amts) else None,
        "outgoing_amount_std": round(float(np.std(out_amts)), 2) if len(out_amts) else None,
        "outgoing_amount_max": round(float(np.max(out_amts)), 2) if len(out_amts) else None,
        "unique_recipients": int(out_df["nameDest"].nunique()) if not out_df.empty else 0,
        "unique_senders": int(in_df["nameOrig"].nunique()) if not in_df.empty else 0,
        "first_step": int(acct_df["step"].min()),
        "last_step": int(acct_df["step"].max()),
        "periodic_hint": bool(periodicity["is_periodic"]),
        "dominant_interval_steps": periodicity["dominant_interval_steps"],
        "type_counts": type_counts,
        "recent_txns": recent_rows,
    }


def query_recipient_history(
    df: pd.DataFrame,
    dest_account: str,
    max_step: int,
) -> dict[str, object]:
    """Summarize strictly past (`step < max_step`) incoming history and fan-in for `dest_account`."""
    past = _strict_past(df, max_step)
    in_df = past[past["nameDest"].astype(str) == str(dest_account)]

    if in_df.empty:
        return {
            "dest_account": str(dest_account),
            "max_step": int(max_step),
            "incoming_count": 0,
            "unique_senders": 0,
            "sender_sample": [],
            "total_incoming_amount": 0.0,
            "mean_incoming_amount": None,
            "max_incoming_amount": None,
            "first_step": None,
            "last_step": None,
            "recent_window_24_count": 0,
            "type_counts": {},
        }

    in_df = in_df.sort_values(["step"], kind="mergesort")
    amts = in_df["amount"].to_numpy(dtype=np.float64)
    unique_senders = sorted(in_df["nameOrig"].astype(str).unique().tolist())
    recent_24 = int((in_df["step"] >= int(max_step) - 24).sum())

    type_counts: dict[str, int] = {}
    if "type" in in_df.columns:
        type_counts = {
            str(k): int(v) for k, v in in_df["type"].value_counts().sort_index().items()
        }

    return {
        "dest_account": str(dest_account),
        "max_step": int(max_step),
        "incoming_count": int(len(in_df)),
        "unique_senders": int(len(unique_senders)),
        "sender_sample": unique_senders[:10],
        "total_incoming_amount": round(float(np.sum(amts)), 2),
        "mean_incoming_amount": round(float(np.mean(amts)), 2),
        "max_incoming_amount": round(float(np.max(amts)), 2),
        "first_step": int(in_df["step"].min()),
        "last_step": int(in_df["step"].max()),
        "recent_window_24_count": recent_24,
        "type_counts": type_counts,
    }


def query_linked_accounts(
    df: pd.DataFrame,
    account_id: str,
    max_step: int,
    max_hops: int = 2,
) -> dict[str, object]:
    """Trace outgoing and incoming multi-hop neighborhood strictly for `step < max_step`.

    Uses hop-by-hop vectorized pandas `.isin()` expansion rather than building a full-table
    Python adjacency dict, keeping per-candidate queries fast on 6M-row PaySim splits.
    """
    past = _strict_past(df, max_step)
    acct = str(account_id)
    hops_limit = max(1, int(max_hops))

    if past.empty:
        return {
            "account_id": acct,
            "max_step": int(max_step),
            "max_hops": hops_limit,
            "outgoing_hops": {},
            "incoming_hops": {},
            "total_outgoing_linked": 0,
            "total_incoming_linked": 0,
            "edges_sample": [],
        }

    orig_series = past["nameOrig"].astype(str)
    dest_series = past["nameDest"].astype(str)

    def _expand_hops(outgoing: bool) -> tuple[dict[str, list[str]], set[str]]:
        visited = {acct}
        frontier = {acct}
        by_hop: dict[str, list[str]] = {}
        for depth in range(hops_limit):
            if not frontier:
                break
            if outgoing:
                raw_neighbors = dest_series[orig_series.isin(frontier)].unique().tolist()
            else:
                raw_neighbors = orig_series[dest_series.isin(frontier)].unique().tolist()

            next_nodes = sorted(set(raw_neighbors) - visited)
            if not next_nodes:
                break
            visited.update(next_nodes)
            by_hop[f"hop_{depth + 1}"] = next_nodes[:25]
            frontier = set(next_nodes)
        return by_hop, visited - {acct}

    out_hops, out_visited = _expand_hops(outgoing=True)
    in_hops, in_visited = _expand_hops(outgoing=False)

    neighborhood = {acct} | out_visited | in_visited
    edge_df = past[
        orig_series.isin(neighborhood) & dest_series.isin(neighborhood)
    ].sort_values(["step", "nameOrig", "nameDest"], kind="mergesort")

    edges_sample: list[dict[str, object]] = []
    for row in edge_df.tail(15).itertuples(index=False):
        edges_sample.append(
            {
                "step": int(row.step),
                "nameOrig": str(row.nameOrig),
                "nameDest": str(row.nameDest),
                "amount": round(float(row.amount), 2),
            }
        )

    return {
        "account_id": acct,
        "max_step": int(max_step),
        "max_hops": hops_limit,
        "outgoing_hops": out_hops,
        "incoming_hops": in_hops,
        "total_outgoing_linked": int(len(out_visited)),
        "total_incoming_linked": int(len(in_visited)),
        "edges_sample": edges_sample,
    }


def fulfill_evidence_request(
    df: pd.DataFrame,
    request_type: str,
    payload: Mapping[str, object],
    max_step: int,
    memory_retriever: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Execute a reviewer follow-up evidence request strictly over `step < max_step`."""
    req_norm = str(request_type).strip().upper()
    if req_norm not in SUPPORTED_EVIDENCE_REQUESTS:
        raise ValueError(
            f"Unsupported evidence request_type '{request_type}'. "
            f"Supported: {sorted(SUPPORTED_EVIDENCE_REQUESTS)}"
        )

    if req_norm == "CHECK_PERIODIC_HISTORY":
        account_id = str(payload.get("account_id") or payload.get("nameOrig") or "")
        if not account_id:
            raise ValueError("CHECK_PERIODIC_HISTORY requires 'account_id' or 'nameOrig' in payload.")
        past = _strict_past(df, max_step)
        acct_df = past[
            (past["nameOrig"].astype(str) == account_id)
            | (past["nameDest"].astype(str) == account_id)
        ].sort_values(["step"], kind="mergesort")

        if acct_df.empty:
            return {
                "request_type": req_norm,
                "account_id": account_id,
                "max_step": int(max_step),
                "is_periodic": False,
                "dominant_interval_steps": None,
                "interval_cv": None,
                "amount_cv": None,
                "repeat_counterparty_ratio": 0.0,
                "prior_cycle_count": 0,
                "summary": f"No prior transactions found for {account_id} before step {max_step}.",
            }

        periodicity = _analyze_periodicity(
            steps=acct_df["step"].to_numpy(dtype=np.int64),
            amounts=acct_df["amount"].to_numpy(dtype=np.float64),
        )
        counterparties = np.where(
            acct_df["nameOrig"].astype(str) == account_id,
            acct_df["nameDest"].astype(str),
            acct_df["nameOrig"].astype(str),
        )
        cp_counts = pd.Series(counterparties).value_counts()
        repeat_cp_ratio = float((cp_counts >= 2).sum() / len(cp_counts)) if len(cp_counts) > 0 else 0.0

        summary = (
            f"Account {account_id} has {periodicity['prior_cycle_count']} prior cycle(s) before "
            f"step {max_step}; is_periodic={periodicity['is_periodic']}, "
            f"dominant_interval={periodicity['dominant_interval_steps']}, "
            f"repeat_counterparty_ratio={repeat_cp_ratio:.2f}."
        )
        return {
            "request_type": req_norm,
            "account_id": account_id,
            "max_step": int(max_step),
            **periodicity,
            "repeat_counterparty_ratio": round(repeat_cp_ratio, 4),
            "summary": summary,
        }

    if req_norm == "QUERY_RECIPIENT_FAN_IN":
        dest_account = str(payload.get("dest_account") or payload.get("nameDest") or "")
        if not dest_account:
            raise ValueError("QUERY_RECIPIENT_FAN_IN requires 'dest_account' or 'nameDest' in payload.")
        result = query_recipient_history(df, dest_account=dest_account, max_step=max_step)
        return {"request_type": req_norm, **result}

    if req_norm == "TRACE_DOWNSTREAM_HOPS":
        # Defaults to `account_id`, or `nameDest` / `nameOrig` if passed explicitly
        account_id = str(
            payload.get("account_id") or payload.get("nameDest") or payload.get("nameOrig") or ""
        )
        if not account_id:
            raise ValueError(
                "TRACE_DOWNSTREAM_HOPS requires 'account_id', 'nameDest', or 'nameOrig' in payload."
            )
        max_hops = int(payload.get("max_hops", 2))
        result = query_linked_accounts(
            df,
            account_id=account_id,
            max_step=max_step,
            max_hops=max_hops,
        )
        return {"request_type": req_norm, **result}

    # FETCH_SIMILAR_FALSE_ALARMS (active only when Setup D supplies a verified memory retriever)
    if memory_retriever is None:
        return {
            "request_type": req_norm,
            "memory_available": False,
            "matches": [],
            "summary": "No verified memory adapter supplied; returning empty match list.",
        }

    raw_matches = memory_retriever(dict(payload))
    matches = list(raw_matches) if raw_matches else []
    return {
        "request_type": req_norm,
        "memory_available": True,
        "matches": matches,
        "summary": f"Retrieved {len(matches)} similar false-alarm record(s) from memory.",
    }


# ---------------------------------------------------------------------------
# Structured Case Generation with Validation & Retry
# ---------------------------------------------------------------------------


def _extract_json_object(raw_text: str) -> dict[str, object]:
    """Parse a JSON object from LLM text, tolerating markdown fences or brief preamble."""
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
        # Fallback: extract the outermost `{ ... }` block if the LLM added prose around JSON
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected top-level JSON object from LLM.")
    return parsed


def _build_fallback_case_fields(
    detector_scores: Mapping[str, object],
    tool_summaries: Mapping[str, object],
) -> dict[str, object]:
    """Deterministic fallback synthesis if the LLM fails schema validation after max_retries."""
    anom = float(detector_scores.get("combined_anomaly_score", 0.0) or 0.0)
    pat = float(detector_scores.get("pattern_score", 0.0) or 0.0)
    conf = round(float(min(max(0.55 * anom + 0.45 * pat, 0.05), 0.99)), 4)

    orig_hist = tool_summaries.get("account_history", {})
    periodic_hint = isinstance(orig_hist, Mapping) and bool(orig_hist.get("periodic_hint", False))
    evidence_for = [f"Fallback synthesis: anomaly={anom:.3f}, pattern={pat:.3f}."]
    evidence_against = (
        ["Prior account history indicates periodic activity."] if periodic_hint else []
    )
    claim = "likely_fraud" if (conf >= 0.55 and not periodic_hint) else "suspicious_requires_review"
    return {
        "claim": claim,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "confidence": conf,
    }


def _extract_detector_scores(candidate: Mapping[str, object]) -> dict[str, object]:
    """Extract numeric and pattern-provenance fields from a ranked candidate row.

    Explicitly excludes `isFraud` so ground-truth labels never enter the case payload.
    """
    score_keys = (
        "combined_anomaly_score",
        "setup_a_gb_prob",
        "setup_a_pred",
        "iso_anomaly_score",
        "pattern_score",
        "pattern_types",
        "pattern_flag_count",
        "pattern_reasons",
        "pattern_event_ids",
        "source",
        "ranking_score",
        "rank",
    )
    scores: dict[str, object] = {}
    for key in score_keys:
        if key in candidate and pd.notna(candidate[key]):
            val = candidate[key]
            if isinstance(val, (np.floating, float)):
                scores[key] = round(float(val), 6)
            elif isinstance(val, (np.integer, int)):
                scores[key] = int(val)
            else:
                scores[key] = str(val)
    return scores


def investigate_candidate(
    candidate_row: Mapping[str, object] | pd.Series,
    history_df: pd.DataFrame,
    llm_client: LLMClient | None = None,
    case_id: str | None = None,
    review_round: int = 1,
    memory_hints: list[dict[str, object]] | None = None,
    extra_tool_summaries: Mapping[str, object] | None = None,
    max_retries: int = 2,
) -> InvestigationCase:
    """Investigate a single candidate transaction using leak-free tools and validated LLM JSON."""
    cand = dict(candidate_row)
    client = llm_client or LLMClient(LLMConfig())

    step = int(cand["step"])
    name_orig = str(cand["nameOrig"])
    name_dest = str(cand["nameDest"])
    txn_id = str(cand.get("txn_id") or f"STEP_{step}_{name_orig}_{name_dest}")
    resolved_case_id = case_id or f"CASE_{txn_id}"
    hints = list(memory_hints) if memory_hints else []

    detector_scores = _extract_detector_scores(cand)
    tool_summaries: dict[str, object] = {
        "account_history": query_account_history(history_df, name_orig, max_step=step),
        "recipient_history": query_recipient_history(history_df, name_dest, max_step=step),
        "linked_accounts": query_linked_accounts(history_df, name_orig, max_step=step, max_hops=2),
    }
    if extra_tool_summaries:
        for k, v in extra_tool_summaries.items():
            tool_summaries[str(k)] = v

    context_payload = {
        "case_id": resolved_case_id,
        "review_round": int(review_round),
        "transaction_summary": {
            "txn_id": txn_id,
            "step": step,
            "type": str(cand.get("type", "UNKNOWN")),
            "amount": round(float(cand["amount"]), 2),
            "nameOrig": name_orig,
            "nameDest": name_dest,
        },
        "detector_scores": detector_scores,
        "tool_summaries": tool_summaries,
        "memory_hints": hints,
    }
    schema_hint = {
        "schema_name": "InvestigationCase",
        "required_keys": ["claim", "evidence_for", "evidence_against", "confidence"],
        "context": context_payload,
    }

    round_instruction = (
        f"This is follow-up review round {review_round}; incorporate the newly gathered "
        "reviewer evidence in `tool_summaries` when updating your claim and confidence.\n"
        if review_round > 1
        else ""
    )
    base_prompt = (
        "Analyze the following bounded investigation tool summaries (all history strictly "
        "precedes the candidate step) and return a JSON object with keys "
        "`claim`, `evidence_for`, `evidence_against`, and `confidence` (float in [0, 1]).\n"
        f"{round_instruction}\n"
        + json.dumps(context_payload, indent=2, sort_keys=True)
    )

    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        step_name = f"investigate_round_{review_round}"
        if attempt > 0:
            step_name = f"investigate_round_{review_round}_retry_{attempt}"
            prompt = (
                f"{base_prompt}\n\nYour previous response failed JSON schema validation with "
                f"errors: {last_errors}. Return ONLY a valid JSON object matching the schema."
            )

        resp = client.generate(
            prompt=prompt,
            system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
            schema_hint=schema_hint,
            case_id=resolved_case_id,
            step_name=step_name,
        )

        try:
            llm_fields = _extract_json_object(resp.content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_errors = [f"JSON parse error: {exc}"]
            continue

        assembled = {
            "case_id": resolved_case_id,
            "review_round": int(review_round),
            "transaction_ids": [txn_id],
            "account_ids": {"nameOrig": name_orig, "nameDest": name_dest},
            "detector_scores": detector_scores,
            "tool_summaries": tool_summaries,
            "claim": llm_fields.get("claim"),
            "evidence_for": llm_fields.get("evidence_for"),
            "evidence_against": llm_fields.get("evidence_against"),
            "confidence": llm_fields.get("confidence"),
            "memory_hints": hints,
            "prompt_version": resp.prompt_version,
            "model_version": resp.model_version,
            "seed": resp.seed,
        }
        val_errors = validate_investigation_case(assembled)
        if not val_errors:
            return InvestigationCase.from_dict(assembled)
        last_errors = val_errors

    # Safe deterministic fallback after exhausting retries
    fallback_fields = _build_fallback_case_fields(detector_scores, tool_summaries)
    tool_summaries["validation_fallback"] = {
        "used": True,
        "retries_attempted": max_retries,
        "last_errors": last_errors,
    }
    return InvestigationCase(
        case_id=resolved_case_id,
        review_round=int(review_round),
        transaction_ids=[txn_id],
        account_ids={"nameOrig": name_orig, "nameDest": name_dest},
        detector_scores=detector_scores,
        tool_summaries=tool_summaries,
        claim=str(fallback_fields["claim"]),
        evidence_for=list(fallback_fields["evidence_for"]),  # type: ignore[arg-type]
        evidence_against=list(fallback_fields["evidence_against"]),  # type: ignore[arg-type]
        confidence=float(fallback_fields["confidence"]),
        memory_hints=hints,
        prompt_version=client.config.prompt_version,
        model_version=client.config.model_version,
        seed=client.config.seed,
    )


def update_case_with_evidence(
    case: InvestigationCase,
    candidate_row: Mapping[str, object] | pd.Series,
    history_df: pd.DataFrame,
    evidence_requests: list[Mapping[str, object] | str],
    llm_client: LLMClient | None = None,
    memory_retriever: Callable[..., Any] | None = None,
    max_retries: int = 2,
) -> InvestigationCase:
    """Fulfill reviewer follow-up evidence requests and re-synthesize the case."""
    cand = dict(candidate_row)
    step = int(cand["step"])
    merged_summaries = dict(case.tool_summaries)

    for idx, req in enumerate(evidence_requests, start=1):
        if isinstance(req, str):
            req_type = req
            payload: dict[str, object] = {
                "account_id": case.account_ids["nameOrig"],
                "nameOrig": case.account_ids["nameOrig"],
                "dest_account": case.account_ids["nameDest"],
                "nameDest": case.account_ids["nameDest"],
            }
        else:
            req_type = str(req.get("request_type") or req.get("type") or "")
            raw_payload = req.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
            payload.setdefault("account_id", case.account_ids["nameOrig"])
            payload.setdefault("nameOrig", case.account_ids["nameOrig"])
            payload.setdefault("dest_account", case.account_ids["nameDest"])
            payload.setdefault("nameDest", case.account_ids["nameDest"])

        ev_result = fulfill_evidence_request(
            df=history_df,
            request_type=req_type,
            payload=payload,
            max_step=step,
            memory_retriever=memory_retriever,
        )
        key = str(req_type).strip().upper()
        if key in merged_summaries:
            key = f"{key}_round_{case.review_round + 1}_{idx}"
        merged_summaries[key] = ev_result

    return investigate_candidate(
        candidate_row=cand,
        history_df=history_df,
        llm_client=llm_client,
        case_id=case.case_id,
        review_round=case.review_round + 1,
        memory_hints=case.memory_hints,
        extra_tool_summaries=merged_summaries,
        max_retries=max_retries,
    )