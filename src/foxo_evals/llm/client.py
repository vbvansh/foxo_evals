"""One LLM client for agents and judges, with live, record and replay modes.

Replay is the default: responses come from recordings/, so tests and evals run
offline, for free, and give the same answer every time. The recording key covers
everything that can change the response (agent, prompt version, model, messages),
so editing a prompt or switching model can never silently reuse an old answer.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from foxo_evals.config import MODE, OPENCODE_BASE_URL, PRICES, RECORDINGS_DIR
from foxo_evals.tracing import Tracer

Mode = Literal["live", "record", "replay"]
T = TypeVar("T", bound=BaseModel)

RETRY_MESSAGE = (
    "Your previous reply did not match the required JSON schema.\n"
    "Validation errors:\n{error}\n"
    "Reply again with only the corrected JSON object."
)


class LLMError(Exception):
    """Base class. The eval runner records these as failures instead of crashing."""


class MissingRecordingError(LLMError):
    pass


class InvalidOutputError(LLMError):
    pass


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            latency_ms=round(self.latency_ms + other.latency_ms, 1),
            cost_usd=self.cost_usd + other.cost_usd,
        )


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 8)


def recording_key(agent: str, prompt_version: str, model: str, messages: list[dict]) -> str:
    blob = json.dumps([agent, prompt_version, model, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence; keep the object only."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def post_to_opencode(payload: dict) -> dict:
    api_key = os.getenv("OPENCODE_API_KEY")
    if not api_key:
        raise LLMError("OPENCODE_API_KEY is not set. Add it to .env, or use replay mode.")
    last_error = ""
    for attempt in range(3):
        if attempt:
            time.sleep(2**attempt)
        try:
            response = httpx.post(
                f"{OPENCODE_BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
        except httpx.TransportError as error:
            last_error = f"network error: {error}"
            continue
        if response.status_code == 200:
            return response.json()
        last_error = f"{response.status_code}: {response.text[:300]}"
        # Only rate limits and server errors are worth retrying.
        if response.status_code != 429 and response.status_code < 500:
            break
    raise LLMError(f"API call failed ({last_error})")


class LLMClient:
    def __init__(
        self,
        model: str,
        tracer: Tracer,
        mode: Mode = MODE,
        recordings_dir: Path = RECORDINGS_DIR,
        post: Callable[[dict], dict] = post_to_opencode,
    ):
        self.model = model
        self.tracer = tracer
        self.mode = mode
        self.recordings_dir = recordings_dir
        self.post = post

    def complete_json(
        self, system: str, user: str, schema: type[T], *, agent: str, prompt_version: str
    ) -> tuple[T, Usage]:
        """Ask for JSON matching `schema`. On a validation error, retry once with the error fed back."""
        system = (
            f"{system}\n\nRespond with a single JSON object that matches this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        usage = Usage()
        with self.tracer.span(
            agent, "llm_call", input=user, prompt_version=prompt_version, model=self.model
        ) as span:
            for _ in range(2):
                text, call_usage = self._chat(messages, agent, prompt_version)
                usage += call_usage
                span.tokens_in, span.tokens_out = usage.tokens_in, usage.tokens_out
                span.cost_usd, span.latency_ms = usage.cost_usd, usage.latency_ms
                try:
                    parsed = schema.model_validate_json(extract_json(text))
                except ValidationError as error:
                    last_error = error
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
                    ]
                    continue
                span.set_output(parsed)
                return parsed, usage
            span.output_summary = text[:400]
            raise InvalidOutputError(
                f"{agent} output did not match {schema.__name__} after one retry: {last_error}"
            ) from last_error

    def _chat(self, messages: list[dict], agent: str, prompt_version: str) -> tuple[str, Usage]:
        key = recording_key(agent, prompt_version, self.model, messages)
        path = self.recordings_dir / agent / f"{key}.json"

        if self.mode == "replay" or (self.mode == "record" and path.exists()):
            if not path.exists():
                raise MissingRecordingError(
                    f"No recording for case '{self.tracer.trace_id}', agent '{agent}' "
                    f"(prompt {prompt_version}, model {self.model}, key {key[:12]}). "
                    "Run in record mode to create it."
                )
            saved = json.loads(path.read_text(encoding="utf-8"))
            return saved["response"], Usage.model_validate(saved["usage"])

        start = time.perf_counter()
        data = self.post(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        text = data["choices"][0]["message"]["content"] or ""
        tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = data.get("usage", {}).get("completion_tokens", 0)
        usage = Usage(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd(self.model, tokens_in, tokens_out),
        )

        if self.mode == "record":
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = {
                "agent": agent,
                "prompt_version": prompt_version,
                "model": self.model,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "messages": messages,
                "response": text,
                "usage": usage.model_dump(),
            }
            path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return text, usage
