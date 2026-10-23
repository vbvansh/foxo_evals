"""Eval registry. Every eval is a small function registered with @eval_check.

A check receives (case, output, trace) and returns a Verdict. The registry turns
that into an EvalResult that carries the case id and the span the check judged,
so a failure in the report links straight to the step that produced it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from foxo_evals.tracing import Span

Kind = Literal["deterministic", "judge"]
Severity = Literal["critical", "major", "minor"]

# Which dataset each agent's evals run on.
DATASET_FOR_AGENT = {"ome": "ome", "mood": "mood", "doctor": "doctor", "coach": "coach", "system": "e2e"}


class EvalResult(BaseModel):
    eval_id: str
    case_id: str
    passed: bool
    score: float | None = None
    details: str = ""
    trace_id: str | None = None
    span_id: str | None = None


@dataclass
class Verdict:
    passed: bool
    details: str = ""
    score: float | None = None
    span: Span | None = None  # set when the failure belongs to a span other than the default


def ok(details: str = "", score: float | None = None) -> Verdict:
    return Verdict(True, details, score)


def fail(details: str, score: float | None = None, span: Span | None = None) -> Verdict:
    return Verdict(False, details, score, span)


def verdict_from(problems: list[str], score: float | None = None) -> Verdict:
    return fail("; ".join(problems), score) if problems else ok(score=score)


def find_span(trace: list[Span], agent: str, step: str) -> Span | None:
    return next((s for s in trace if s.agent == agent and s.step == step), None)


@dataclass
class EvalSpec:
    id: str
    agent: str
    kind: Kind
    severity: Severity
    description: str
    check: Callable[[Any, Any, list[Span]], Verdict]
    applies: Callable[[Any], bool]
    step: str
    threshold: float
    # Judge evals only: which judge prompt, and the function that builds what it reads.
    judge: str | None = None
    subject: Callable[[Any, Any], str] | None = None

    @property
    def dataset(self) -> str:
        return DATASET_FOR_AGENT[self.agent]

    def run(self, case: Any, output: Any, trace: list[Span]) -> EvalResult:
        verdict = self.check(case, output, trace)
        # Point at the span this eval judges, e.g. Doctor -> llm_call. System
        # evals have no agent span of their own, so they point at the journey root.
        span = verdict.span or find_span(trace, self.agent, self.step) or (trace[0] if trace else None)
        return EvalResult(
            eval_id=self.id,
            case_id=case.case_id,
            passed=verdict.passed,
            score=verdict.score,
            details=verdict.details,
            trace_id=case.case_id,
            span_id=span.span_id if span else None,
        )


REGISTRY: dict[str, EvalSpec] = {}


def eval_check(
    *,
    id: str,
    agent: str,
    kind: Kind,
    severity: Severity,
    description: str,
    applies: Callable[[Any], bool] = lambda case: True,
    step: str = "llm_call",
    threshold: float = 0.9,
) -> Callable:
    """Register a check. `applies` picks the cases it runs on; `threshold` matters for major evals."""

    def register(check: Callable) -> Callable:
        if id in REGISTRY:
            raise ValueError(f"duplicate eval id {id}")
        REGISTRY[id] = EvalSpec(id, agent, kind, severity, description, check, applies, step, threshold)
        return check

    return register


def tagged(tag: str) -> Callable[[Any], bool]:
    return lambda case: tag in case.tags
