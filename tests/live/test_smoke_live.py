"""Live smoke test: one full journey against the real API.

Skipped by default. Run with: python -m pytest -m live -s
"""

import pytest

from foxo_evals.config import AGENT_MODEL
from foxo_evals.llm.client import LLMClient
from foxo_evals.system.context import AgentContext
from foxo_evals.system.orchestrator import run_member_journey
from foxo_evals.system.schemas import JourneyInput
from foxo_evals.tracing import Tracer, format_trace

REPORT = """FOXO DIAGNOSTICS - LIPID & METABOLIC PANEL
Patient: Rahul Mehta   Age/Sex: 46/M   Collected: 18-Oct-2026 07:40
Test                     Result    Units     Ref. Range
Glucose (Fasting)        5.9       mmol/L    3.9 - 5.5
HbA1c                    5.8       %         4.0 - 5.6
Total Cholesterol        236       mg/dL     < 200
LDL Cholesterol          158       mg/dL     < 100
HDL Cholesterol          41        mg/dL     > 40
Triglycerides            182       mg/dL     < 150
Remarks: Sample received in good condition.
"""


@pytest.mark.live
def test_one_routine_journey_runs_live(tmp_path):
    tracer = Tracer(tmp_path / "traces.jsonl")
    ctx = AgentContext(llm=LLMClient(AGENT_MODEL, tracer, mode="live"), tracer=tracer)
    journey = JourneyInput.model_validate(
        {
            "message": {"member_id": "M100", "text": "My report came today. Can you tell me what it means?"},
            "profile": {
                "member_id": "M100", "full_name": "Rahul Mehta", "age": 46, "sex": "male",
                "diet": "vegetarian", "conditions": [], "medications": [],
            },
            "lab_report": {
                "report_id": "R100", "member_id": "M100", "member_sex": "male", "member_age": 46,
                "collected_at": "2026-10-18", "raw_text": REPORT,
            },
        }
    )
    with tracer.trace("SMOKE-01"):
        result = run_member_journey(journey, ctx)

    print(format_trace(tracer.spans))
    print(result.model_dump_json(indent=2))
    assert result.error is None
    assert result.trajectory == ["ome", "mood", "doctor", "coach"]
