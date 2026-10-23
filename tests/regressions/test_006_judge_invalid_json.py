"""SEEDED: a judge that returns invalid JSON must fail its eval, not crash the run.

This failure was not hit while building; it is seeded because judges are the least
predictable part of the harness.
Symptom (seeded): the judge model answers with prose instead of the JSON verdict, twice.
Where in the trace: <case> -> judges/<name> -> llm_call, status "error".
Root cause: model output does not match JudgeOutput, and the one retry fails too.
Fix: the client raises InvalidOutputError (an LLMError); the runner records the eval
as failed with "judge call failed" and carries on with the remaining evals.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals import judge
from foxo_evals.llm.client import LLMClient
from foxo_evals.runner import REGISTRY, CaseRun, evaluate
from foxo_evals.system.schemas import CoachPlan
from foxo_evals.tracing import Tracer


def prose_every_time(payload: dict, session_id: str) -> dict:
    return {"choices": [{"message": {"content": "The plan looks good to me overall."}}], "usage": {}}


def test_invalid_judge_json_is_a_failed_eval(tmp_path, monkeypatch):
    tracer = Tracer()
    judge.set_client(LLMClient("glm-5.3", tracer, mode="live", recordings_dir=tmp_path, post=prose_every_time))
    monkeypatch.setattr("foxo_evals.runner.REGISTRY", {"COACH-05": REGISTRY["COACH-05"]})
    case = load_dataset("coach")[0]
    plan = CoachPlan(focus_areas=["LDL"], diet_recommendations=[], activity="Walk.", sleep="7 hours", retest_in_weeks=12)
    try:
        [result] = evaluate([CaseRun("coach", case, plan, None)], tracer)
    finally:
        judge.set_client(None)
    assert not result.passed
    assert result.details.startswith("judge call failed: InvalidOutputError")
    [span] = [s for s in tracer.spans if s.agent == "judges/coach_actionability"]
    assert span.status == "error" and span.trace_id == case.case_id
