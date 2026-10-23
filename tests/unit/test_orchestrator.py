from foxo_evals.llm.client import InvalidOutputError, Usage
from foxo_evals.system.context import AgentContext
from foxo_evals.system.orchestrator import run_member_journey
from foxo_evals.system.schemas import JourneyInput
from foxo_evals.tracing import Tracer


class ScriptedLLM:
    """Returns a fixed output per agent, so routing can be tested without any model."""

    def __init__(self, outputs: dict):
        self.outputs = outputs
        self.calls: list[str] = []

    def complete_json(self, system, user, schema, *, agent, prompt_version):
        self.calls.append(agent)
        output = self.outputs[agent]
        if isinstance(output, Exception):
            raise output
        return schema.model_validate(output), Usage()


def ome_output(handoff_to: str, intent: str = "report_question", escalated: bool = False) -> dict:
    return {
        "intent": intent,
        "urgency": "emergency" if escalated else "routine",
        "reply_text": "A FOXO doctor will review this.",
        "handoff_to": handoff_to,
        "escalated": escalated,
        "escalation_reason": "chest pain" if escalated else None,
    }


MOOD_OUTPUT = {"biomarkers": [{"name_as_printed": "LDL", "value_as_printed": "160", "unit_as_printed": "mg/dL"}]}
DOCTOR_OUTPUT = {
    "summary_for_clinician": "LDL 160 mg/dL is above range.",
    "abnormal_findings": [{"biomarker": "ldl", "value": 160, "unit": "mg/dL", "flag": "high", "comment": ""}],
    "tool_calls": [{"tool": "book_appointment", "specialty": "cardiology", "urgency": "routine", "reason": "LDL"}],
}
COACH_OUTPUT = {
    "focus_areas": ["LDL"],
    "diet_recommendations": [{"item": "oats", "rationale": "fibre", "linked_biomarker": "ldl"}],
    "activity": "walk",
    "sleep": "7 hours",
    "retest_in_weeks": 12,
}


def journey(with_report: bool = True) -> JourneyInput:
    report = {
        "report_id": "R1", "member_id": "M1", "member_sex": "male", "member_age": 45,
        "collected_at": "2026-10-20", "raw_text": "LDL 160 mg/dL",
    }
    return JourneyInput.model_validate(
        {
            "message": {"member_id": "M1", "text": "Can you explain my report?"},
            "profile": {"member_id": "M1", "age": 45, "sex": "male", "diet": "vegetarian"},
            "lab_report": report if with_report else None,
        }
    )


def run(outputs: dict, case: JourneyInput):
    tracer = Tracer()
    llm = ScriptedLLM(outputs)
    with tracer.trace("CASE-1"):
        result = run_member_journey(case, AgentContext(llm=llm, tracer=tracer))
    return result, tracer, llm


def test_report_question_runs_all_four_agents():
    outputs = {"ome": ome_output("doctor"), "mood": MOOD_OUTPUT, "doctor": DOCTOR_OUTPUT, "coach": COACH_OUTPUT}
    result, tracer, _ = run(outputs, journey())
    assert result.trajectory == ["ome", "mood", "doctor", "coach"]
    assert result.lab_result.biomarkers[0].flag == "high"
    assert result.doctor.appointments[0].member_id == "M1"
    steps = [s.step for s in tracer.spans]
    assert {"handoff:mood", "handoff:doctor", "handoff:coach", "tool:book_appointment"} <= set(steps)


def test_emergency_stops_after_ome():
    result, tracer, llm = run({"ome": ome_output("human_clinician", "symptom_concern", True)}, journey())
    assert result.trajectory == ["ome"]
    assert llm.calls == ["ome"]
    assert "tool:escalate_to_human" in [s.step for s in tracer.spans]


def test_critical_alert_skips_the_coach():
    doctor = {**DOCTOR_OUTPUT, "critical_alerts": ["potassium"]}
    outputs = {"ome": ome_output("doctor"), "mood": MOOD_OUTPUT, "doctor": doctor}
    result, _, _ = run(outputs, journey())
    assert result.trajectory == ["ome", "mood", "doctor"]
    assert result.coach is None


def test_prescription_request_without_report_goes_to_doctor_only():
    outputs = {"ome": ome_output("doctor", "prescription_request"), "doctor": DOCTOR_OUTPUT}
    result, _, _ = run(outputs, journey(with_report=False))
    assert result.trajectory == ["ome", "doctor"]


def test_agent_failure_is_recorded_not_raised():
    outputs = {"ome": ome_output("doctor"), "mood": MOOD_OUTPUT, "doctor": InvalidOutputError("bad JSON")}
    result, tracer, _ = run(outputs, journey())
    assert result.trajectory == ["ome", "mood"]
    assert result.error == "InvalidOutputError: bad JSON"
    doctor_span = next(s for s in tracer.spans if s.agent == "doctor" and s.step == "agent")
    assert doctor_span.status == "error"
