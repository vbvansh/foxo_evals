"""Checks on whole journeys: handoffs, information passed between agents, routing, latency, cost."""

from foxo_evals.config import JOURNEY_COST_BUDGET_USD, JOURNEY_LATENCY_BUDGET_MS
from foxo_evals.datasets.schemas import E2ECase
from foxo_evals.evals.registry import Verdict, eval_check, fail, find_span, ok, verdict_from
from foxo_evals.evals.text import mentions
from foxo_evals.system.schemas import JourneyResult
from foxo_evals.tracing import Span


def llm_spans(trace: list[Span]) -> list[Span]:
    return [s for s in trace if s.step == "llm_call"]


def journey_latency_ms(trace: list[Span]) -> float:
    """Time spent in model calls. In replay mode these are the latencies measured at record time."""
    return sum(s.latency_ms or 0 for s in llm_spans(trace))


def journey_cost_usd(trace: list[Span]) -> float:
    return sum(s.cost_usd for s in llm_spans(trace))


@eval_check(
    id="SYS-01", agent="system", kind="deterministic", severity="critical",
    description="Every handoff object between agents is schema-valid",
)
def handoffs_valid(case: E2ECase, output: JourneyResult, trace: list[Span]) -> Verdict:
    broken = [s for s in trace if s.step.startswith("handoff:") and s.status != "ok"]
    if broken:
        return fail(f"{broken[0].step} failed: {broken[0].error}", span=broken[0])
    if output.error and "ValidationError" in output.error:
        return fail(output.error)
    return ok()


@eval_check(
    id="SYS-02", agent="system", kind="deterministic", severity="critical",
    description="Every critical value flagged by Mood reaches the Doctor's critical_alerts",
    applies=lambda case: bool(case.gold.critical),
)
def critical_values_preserved(case: E2ECase, output: JourneyResult, trace: list[Span]) -> Verdict:
    doctor_span = find_span(trace, "doctor", "llm_call")
    if output.lab_result is None or output.doctor is None:
        return fail(f"journey stopped before the Doctor reviewed the report: {output.trajectory}")
    flagged = {b.name for b in output.lab_result.biomarkers if b.flag.startswith("critical")}
    lost = [
        name for name in sorted(flagged | set(case.gold.critical))
        if not any(mentions(alert, name) for alert in output.doctor.critical_alerts)
    ]
    if lost:
        return fail(f"critical values lost between Mood and Doctor: {lost}", span=doctor_span)
    return ok()


@eval_check(
    id="SYS-03", agent="system", kind="deterministic", severity="major",
    description="The sequence of agents matches the gold path",
)
def trajectory_matches(case: E2ECase, output: JourneyResult, trace: list[Span]) -> Verdict:
    problems = []
    if output.trajectory != case.gold.trajectory:
        problems.append(f"expected {' > '.join(case.gold.trajectory)}, got {' > '.join(output.trajectory) or 'nothing'}")
    if case.gold.escalate and not (output.ome and output.ome.escalated):
        problems.append("emergency was not escalated")
    if output.error:
        problems.append(f"journey error: {output.error}")
    return verdict_from(problems)


@eval_check(
    id="SYS-04", agent="system", kind="deterministic", severity="minor",
    description=f"Model time per journey within {JOURNEY_LATENCY_BUDGET_MS / 1000:.0f} s",
)
def latency_within_budget(case: E2ECase, output: JourneyResult, trace: list[Span]) -> Verdict:
    latency = journey_latency_ms(trace)
    details = f"{latency / 1000:.1f} s"
    score = round(latency / 1000, 1)
    return ok(details, score) if latency <= JOURNEY_LATENCY_BUDGET_MS else fail(details, score)


@eval_check(
    id="SYS-05", agent="system", kind="deterministic", severity="minor",
    description=f"Cost per journey within ${JOURNEY_COST_BUDGET_USD}",
)
def cost_within_budget(case: E2ECase, output: JourneyResult, trace: list[Span]) -> Verdict:
    cost = journey_cost_usd(trace)
    details = f"${cost:.5f}"
    return ok(details, cost) if cost <= JOURNEY_COST_BUDGET_USD else fail(details, cost)
