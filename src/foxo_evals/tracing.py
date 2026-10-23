"""A tiny tracer. Every agent step, LLM call and tool call becomes one span.

Spans are written to runs/<run_id>/traces.jsonl, one per line, so a failing eval
can point at the exact step that broke: "Doctor -> llm_call (span 3f9a1c02)".
"""

import hashlib
import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

PII_FIELDS = {"full_name", "phone"}
PHONE_PATTERN = re.compile(r"(\+91[\s-]?)?\b\d{5}\s?\d{5}\b")
SUMMARY_LIMIT = 400


class Span(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    agent: str
    step: str
    prompt_version: str | None = None
    model: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    # LLM calls only: the recordings/ files this call read or wrote.
    recording_keys: list[str] = []

    def set_output(self, value: Any) -> None:
        self.output_summary = summarise(value)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[redacted]" if k in PII_FIELDS and v else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return PHONE_PATTERN.sub("[phone]", value)
    return value


def summarise(value: Any) -> str:
    """Short, PII-redacted text version of any input or output, for the trace."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    value = redact(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= SUMMARY_LIMIT else text[:SUMMARY_LIMIT] + "..."


class Tracer:
    """Collects spans for one run. Pass a path to also write them to disk."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.spans: list[Span] = []
        self.trace_id = "untraced"
        self._stack: list[Span] = []
        self._counts: dict[str, int] = {}

    @contextmanager
    def trace(self, trace_id: str) -> Iterator[None]:
        """One trace per case. Spans are written when the block ends, in start order.

        A trace can be reopened (judges add spans after the agents ran); span
        numbering continues where it left off, so ids never repeat.
        """
        self.trace_id = trace_id
        first = len(self.spans)
        try:
            yield
        finally:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    for span in self.spans[first:]:
                        f.write(span.model_dump_json() + "\n")

    @contextmanager
    def span(
        self,
        agent: str,
        step: str,
        *,
        input: Any = None,
        prompt_version: str | None = None,
        model: str | None = None,
    ) -> Iterator[Span]:
        # Span ids are derived from the trace id and position, so replaying the
        # same run gives the same ids and two runs can be compared line by line.
        count = self._counts[self.trace_id] = self._counts.get(self.trace_id, 0) + 1
        span_id = hashlib.sha1(f"{self.trace_id}:{count}".encode()).hexdigest()[:8]
        span = Span(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self._stack[-1].span_id if self._stack else None,
            agent=agent,
            step=step,
            prompt_version=prompt_version,
            model=model,
            input_summary=summarise(input) if input is not None else "",
        )
        self.spans.append(span)
        self._stack.append(span)
        start = time.perf_counter()
        try:
            yield span
        except Exception as error:
            span.status = "error"
            span.error = f"{type(error).__name__}: {error}"[:SUMMARY_LIMIT]
            raise
        finally:
            # LLM calls set latency themselves (the recorded API latency in replay mode).
            if span.latency_ms is None:
                span.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            self._stack.pop()

    def spans_for(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]


def load_spans(path: Path) -> list[Span]:
    return [Span.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def format_trace(spans: list[Span]) -> str:
    """Render one trace as an indented tree, children under their parent."""
    children: dict[str | None, list[Span]] = {}
    for span in spans:
        children.setdefault(span.parent_span_id, []).append(span)

    lines = []

    def walk(parent_id: str | None, depth: int) -> None:
        for span in children.get(parent_id, []):
            parts = [f"{'  ' * depth}[{span.span_id}] {span.agent} > {span.step}", span.status]
            parts.append(f"{span.latency_ms:.0f} ms")
            if span.tokens_in or span.tokens_out:
                parts.append(f"{span.tokens_in}/{span.tokens_out} tok  ${span.cost_usd:.5f}")
            if span.model:
                parts.append(f"{span.model} prompt {span.prompt_version}")
            lines.append("  ".join(parts))
            if span.error:
                lines.append(f"{'  ' * depth}    error: {span.error}")
            walk(span.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)
