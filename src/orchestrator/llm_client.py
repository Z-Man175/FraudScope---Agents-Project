"""Deterministic LLM interface, fallback chain, and JSONL audit logger for FraudScope.

WHAT IT DOES
    Wraps all LLM calls behind a single interface (`LLMClient.generate`) that:
    1. Enforces reproducible generation parameters (`temperature=0.0`, `top_p=1.0`, `seed=42`).
    2. Executes a 3-tier fallback chain: local HTTP endpoint -> cloud API fallback ->
       offline deterministic stub (so tests and CI run without external API keys).
    3. Logs every call (hashes, latency, token estimates, cost, fallback errors) to
       `outputs/logs/llm_calls.jsonl` for auditing and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42


@dataclass
class LLMConfig:
    model_version: str = "local-deterministic-v1"
    prompt_version: str = "v1"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = RANDOM_SEED
    local_endpoint: str | None = None
    fallback_api_endpoint: str | None = None
    fallback_api_key_env: str = "FRAUDSCOPE_LLM_API_KEY"
    allow_deterministic_stub: bool = True
    timeout_seconds: float = 30.0
    cost_per_1k_input_tokens_usd: float = 0.0
    cost_per_1k_output_tokens_usd: float = 0.0
    log_path: Path = REPO_ROOT / "outputs" / "logs" / "llm_calls.jsonl"


@dataclass
class LLMResponse:
    content: str
    mode: str
    model_version: str
    prompt_version: str
    seed: int
    latency_ms: float
    input_tokens_est: int
    output_tokens_est: int
    estimated_cost_usd: float
    input_hash: str
    output_hash: str
    case_id: str | None = None
    step_name: str = "unknown"
    fallback_reason: str | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Fast ~4 chars/token heuristic when backend token counts are unavailable."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class LLMClient:
    """Single LLM wrapper with deterministic settings, fallback handling, and audit logging."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.call_count: int = 0
        self.total_input_tokens_est: int = 0
        self.total_output_tokens_est: int = 0
        self.total_estimated_cost_usd: float = 0.0
        self.total_latency_ms: float = 0.0

    def _call_http_endpoint(
        self,
        endpoint: str,
        prompt: str,
        system_prompt: str,
        api_key: str | None = None,
    ) -> str:
        """POST an OpenAI-compatible chat payload and extract assistant text."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_version,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(first.get("text"), str):
                        return first["text"]
            if isinstance(parsed.get("response"), str):
                return parsed["response"]
            if isinstance(parsed.get("content"), str):
                return parsed["content"]
        raise ValueError("Unexpected LLM HTTP response format.")

    def _deterministic_stub_generate(
        self,
        prompt: str,
        system_prompt: str,
        schema_hint: dict | None,
    ) -> str:
        """Offline rule-based JSON synthesizer grounded strictly in bounded tool summaries."""
        if isinstance(schema_hint, dict) and schema_hint.get("schema_name") == "InvestigationCase":
            context = schema_hint.get("context", {})
            detector_scores = context.get("detector_scores", {}) if isinstance(context, dict) else {}
            tool_summaries = context.get("tool_summaries", {}) if isinstance(context, dict) else {}
            memory_hints = context.get("memory_hints", []) if isinstance(context, dict) else []

            anomaly_score = float(detector_scores.get("combined_anomaly_score", 0.0) or 0.0)
            pattern_score = float(detector_scores.get("pattern_score", 0.0) or 0.0)
            pattern_types = str(detector_scores.get("pattern_types", "") or "")

            orig_hist = tool_summaries.get("account_history", {}) if isinstance(tool_summaries, dict) else {}
            recip_hist = tool_summaries.get("recipient_history", {}) if isinstance(tool_summaries, dict) else {}
            linked_hist = tool_summaries.get("linked_accounts", {}) if isinstance(tool_summaries, dict) else {}
            periodic_check = tool_summaries.get("CHECK_PERIODIC_HISTORY", {}) if isinstance(tool_summaries, dict) else {}

            evidence_for: list[str] = []
            evidence_against: list[str] = []

            if anomaly_score >= 0.7:
                evidence_for.append(
                    f"Elevated anomaly score ({anomaly_score:.3f}) from anomaly detection baseline."
                )
            elif anomaly_score > 0.0:
                evidence_against.append(
                    f"Moderate/low anomaly score ({anomaly_score:.3f}) relative to high-risk alerts."
                )

            if pattern_score > 0.0:
                label = pattern_types or "behavioral_pattern"
                evidence_for.append(
                    f"Pattern detector flagged [{label}] with pattern_score={pattern_score:.3f}."
                )
            else:
                evidence_against.append("No multi-transaction behavioral pattern flag triggered.")

            fan_in = int(recip_hist.get("unique_senders", 0) or 0)
            if fan_in >= 3:
                evidence_for.append(
                    f"Recipient shows elevated prior fan-in ({fan_in} unique senders before current step)."
                )

            out_linked = int(linked_hist.get("total_outgoing_linked", 0) or 0)
            if out_linked >= 2:
                evidence_for.append(
                    f"Origin has {out_linked} prior downstream linked accounts within 2 hops."
                )

            prior_txns = int(orig_hist.get("prior_txn_count", 0) or 0)
            is_periodic = bool(
                orig_hist.get("periodic_hint", False)
                or periodic_check.get("is_periodic", False)
            )
            if is_periodic:
                evidence_against.append(
                    "Prior account history exhibits consistent periodic transfer cadence."
                )
            elif prior_txns >= 3:
                evidence_against.append(
                    f"Origin has established prior history ({prior_txns} prior transactions)."
                )
            else:
                evidence_for.append(
                    f"Origin has limited prior history ({prior_txns} prior transactions before this step)."
                )

            if memory_hints:
                fp_hints = [
                    h for h in memory_hints
                    if isinstance(h, dict) and str(h.get("final_label", "")).lower() in {"0", "false_positive", "legit"}
                ]
                fraud_hints = [
                    h for h in memory_hints
                    if isinstance(h, dict) and str(h.get("final_label", "")).lower() in {"1", "fraud"}
                ]
                if fp_hints:
                    evidence_against.append(
                        f"Verified memory retrieved {len(fp_hints)} similar confirmed false-alarm case(s)."
                    )
                if fraud_hints:
                    evidence_for.append(
                        f"Verified memory retrieved {len(fraud_hints)} similar confirmed fraud case(s)."
                    )

            raw_conf = 0.55 * anomaly_score + 0.45 * pattern_score
            if is_periodic:
                raw_conf -= 0.25
            confidence = round(float(min(max(raw_conf, 0.05), 0.99)), 4)

            if confidence >= 0.55 and not is_periodic:
                claim = "likely_fraud"
            elif is_periodic and confidence < 0.65:
                claim = "likely_legitimate_periodic"
            else:
                claim = "suspicious_requires_review"

            return json.dumps(
                {
                    "claim": claim,
                    "evidence_for": evidence_for,
                    "evidence_against": evidence_against,
                    "confidence": confidence,
                },
                sort_keys=True,
            )

        digest = _sha256_text(f"{system_prompt}|{prompt}|{self.config.seed}")[:12]
        return json.dumps(
            {
                "status": "ok",
                "summary": f"deterministic_stub_response_{digest}",
                "seed": self.config.seed,
            },
            sort_keys=True,
        )

    def _append_log(self, response: LLMResponse) -> None:
        log_file = Path(self.config.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "call_index": self.call_count,
            "case_id": response.case_id,
            "step_name": response.step_name,
            "mode": response.mode,
            "fallback_reason": response.fallback_reason,
            "model_version": response.model_version,
            "prompt_version": response.prompt_version,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": response.seed,
            "latency_ms": round(response.latency_ms, 3),
            "input_tokens_est": response.input_tokens_est,
            "output_tokens_est": response.output_tokens_est,
            "estimated_cost_usd": round(response.estimated_cost_usd, 8),
            "input_hash": response.input_hash,
            "output_hash": response.output_hash,
        }
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: dict | None = None,
        case_id: str | None = None,
        step_name: str = "unknown",
    ) -> LLMResponse:
        """Invoke local LLM -> API fallback -> deterministic stub and log audit record."""
        t0 = time.perf_counter()
        content: str | None = None
        mode = "deterministic_stub"
        errors: list[str] = []

        if self.config.local_endpoint:
            try:
                content = self._call_http_endpoint(
                    endpoint=self.config.local_endpoint,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                mode = "local_http"
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                errors.append(f"local_http: {exc}")

        if content is None and self.config.fallback_api_endpoint:
            api_key = os.environ.get(self.config.fallback_api_key_env)
            try:
                content = self._call_http_endpoint(
                    endpoint=self.config.fallback_api_endpoint,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    api_key=api_key,
                )
                mode = "fallback_api"
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                errors.append(f"fallback_api: {exc}")

        if content is None:
            if not self.config.allow_deterministic_stub:
                raise RuntimeError(
                    "LLM generation failed across configured backends and "
                    f"allow_deterministic_stub=False. Errors: {errors}"
                )
            content = self._deterministic_stub_generate(prompt, system_prompt, schema_hint)
            mode = "deterministic_stub"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        in_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(prompt)
        out_tokens = _estimate_tokens(content)
        cost_usd = (
            (in_tokens / 1000.0) * self.config.cost_per_1k_input_tokens_usd
            + (out_tokens / 1000.0) * self.config.cost_per_1k_output_tokens_usd
        )
        in_hash = _sha256_text(f"{system_prompt}\n---\n{prompt}")
        out_hash = _sha256_text(content)

        self.call_count += 1
        self.total_input_tokens_est += in_tokens
        self.total_output_tokens_est += out_tokens
        self.total_estimated_cost_usd += cost_usd
        self.total_latency_ms += latency_ms

        response = LLMResponse(
            content=content,
            mode=mode,
            model_version=self.config.model_version,
            prompt_version=self.config.prompt_version,
            seed=self.config.seed,
            latency_ms=latency_ms,
            input_tokens_est=in_tokens,
            output_tokens_est=out_tokens,
            estimated_cost_usd=cost_usd,
            input_hash=in_hash,
            output_hash=out_hash,
            case_id=case_id,
            step_name=step_name,
            fallback_reason="; ".join(errors) if errors else None,
        )
        self._append_log(response)
        return response

    def get_usage_summary(self) -> dict[str, object]:
        return {
            "call_count": self.call_count,
            "total_input_tokens_est": self.total_input_tokens_est,
            "total_output_tokens_est": self.total_output_tokens_est,
            "total_estimated_cost_usd": round(self.total_estimated_cost_usd, 8),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "model_version": self.config.model_version,
            "prompt_version": self.config.prompt_version,
            "seed": self.config.seed,
        }