from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals import system_evals as system
from foxo_evals.system.normalise import normalise_report
from foxo_evals.system.schemas import DoctorOutput, JourneyResult, OmeResponse, RawBiomarker
from foxo_evals.tracing import Span

CASES = {c.case_id: c for c in load_dataset("e2e")}


def span(step: str, agent: str = "orchestrator", status: str = "ok", latency: float = 0, cost: float = 0) -> Span:
    return Span(trace_id="T", span_id=f"{agent}-{step}", parent_span_id=None, agent=agent, step=step,
                status=status, latency_ms=latency, cost_usd=cost)


def critical_journey(alerts: list[str]) -> JourneyResult:
    case = CASES["E2E-002"]
    lab = normalise_report(case.journey.lab_report, [
        RawBiomarker(name_as_printed="Potassium", value_as_printed="6.9", unit_as_printed="mmol/L"),
    ])
    ome = OmeResponse(intent="report_question", urgency="routine", reply_text="A doctor will review.",
                      handoff_to="doctor", escalated=False)
    doctor = DoctorOutput(summary_for_clinician="Potassium 6.9 mmol/L is critical.", critical_alerts=alerts)
    return JourneyResult(trajectory=["ome", "mood", "doctor"], ome=ome, lab_result=lab, doctor=doctor)


def test_handoffs_must_be_valid():
    case = CASES["E2E-002"]
    assert system.handoffs_valid(case, critical_journey(["potassium"]), [span("handoff:doctor")]).passed
    broken = span("handoff:doctor", status="error")
    result = system.handoffs_valid(case, critical_journey(["potassium"]), [broken])
    assert not result.passed and result.span is broken


def test_critical_values_must_reach_the_doctor():
    case = CASES["E2E-002"]
    assert system.critical_values_preserved(case, critical_journey(["Potassium 6.9 mmol/L"]), []).passed
    result = system.critical_values_preserved(case, critical_journey([]), [])
    assert not result.passed and "potassium" in result.details


def test_trajectory_must_match_gold():
    case = CASES["E2E-002"]
    assert system.trajectory_matches(case, critical_journey(["potassium"]), []).passed
    too_far = critical_journey(["potassium"]).model_copy(update={"trajectory": ["ome", "mood", "doctor", "coach"]})
    assert not system.trajectory_matches(case, too_far, []).passed


def test_latency_and_cost_budgets():
    case = CASES["E2E-002"]
    cheap = [span("llm_call", "ome", latency=20_000, cost=0.001), span("llm_call", "doctor", latency=40_000, cost=0.002)]
    assert system.latency_within_budget(case, JourneyResult(), cheap).passed
    assert system.cost_within_budget(case, JourneyResult(), cheap).passed
    slow = [span("llm_call", "doctor", latency=400_000, cost=0.5)]
    assert not system.latency_within_budget(case, JourneyResult(), slow).passed
    assert not system.cost_within_budget(case, JourneyResult(), slow).passed
