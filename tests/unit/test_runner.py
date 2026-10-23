import json

import pytest

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.registry import REGISTRY, EvalResult, EvalSpec, eval_check
from foxo_evals.llm.client import InvalidOutputError, Usage
from foxo_evals.runner import CaseRun, eval_stats, evaluate, gate, parse_prompt_versions, percentile, run_evals
from foxo_evals.tracing import Tracer

GENERIC_OUTPUTS = {
    "ome": {"intent": "support", "urgency": "routine", "reply_text": "Happy to help.", "handoff_to": "none",
            "escalated": False},
    "mood": {"biomarkers": []},
    "doctor": {"summary_for_clinician": "Reviewed."},
    "coach": {"focus_areas": [], "diet_recommendations": [], "activity": "Walk.", "sleep": "7 hours",
              "retest_in_weeks": 12},
}


class ScriptedLLM:
    def __init__(self, failing_agent: str | None = None):
        self.failing_agent = failing_agent

    def complete_json(self, system, user, schema, *, agent, prompt_version):
        if agent == self.failing_agent:
            raise InvalidOutputError("bad JSON twice")
        return schema.model_validate(GENERIC_OUTPUTS[agent]), Usage()


def test_parse_prompt_versions():
    assert parse_prompt_versions("v1") == {"ome": "v1", "mood": "v1", "doctor": "v1", "coach": "v1"}
    assert parse_prompt_versions("v1,doctor=v2")["doctor"] == "v2"
    with pytest.raises(ValueError):
        parse_prompt_versions("pharmacist=v1")


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([5], 50) == 5
    assert percentile([], 95) == 0.0


def results(eval_id: str, passed: int, failed: int) -> list[EvalResult]:
    return [EvalResult(eval_id=eval_id, case_id=f"C{i}", passed=i < passed) for i in range(passed + failed)]


def test_gate_blocks_on_any_critical_failure():
    stats = eval_stats(results("DOC-03", 9, 1))
    assert not gate(stats)["passed"]


def test_gate_blocks_major_evals_below_threshold_only():
    assert gate(eval_stats(results("OME-01", 9, 1)))["passed"]  # 90% meets the 90% threshold
    assert not gate(eval_stats(results("OME-01", 8, 2)))["passed"]


def test_minor_failures_never_block():
    stats = eval_stats(results("SYS-04", 0, 3))
    assert stats["SYS-04"]["status"] == "tracked"
    assert gate(stats)["passed"]


def test_eval_ids_are_unique():
    with pytest.raises(ValueError, match="duplicate"):
        eval_check(id="OME-01", agent="ome", kind="deterministic", severity="major", description="x")(lambda *a: None)


def test_crashing_check_becomes_a_failed_result(monkeypatch):
    def broken(case, output, trace):
        raise KeyError("oops")

    spec = EvalSpec("TEST-01", "coach", "deterministic", "minor", "broken check", broken, lambda c: True, "llm_call", 0.9)
    monkeypatch.setattr("foxo_evals.runner.REGISTRY", {"TEST-01": spec})
    case = load_dataset("coach")[0]
    [result] = evaluate([CaseRun("coach", case, object(), None)], Tracer())
    assert not result.passed and "check crashed: KeyError" in result.details


def test_full_run_writes_every_file_and_records_agent_failures(tmp_path):
    out = tmp_path / "run"
    summary = run_evals(out, llm=ScriptedLLM(failing_agent="doctor"))

    for name in ("traces.jsonl", "outputs.jsonl", "results.jsonl", "summary.json", "report.md"):
        assert (out / name).exists(), name
    assert summary["gate"]["passed"] is False
    assert set(summary["evals"]) == set(REGISTRY)

    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    doctor_rows = [r for r in rows if r["eval_id"] == "DOC-01"]
    assert doctor_rows and all("agent produced no output" in r["details"] for r in doctor_rows)
    assert all(r["span_id"] for r in doctor_rows)  # failures still point at the span that broke

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Gate: FAIL" in report and "doctor → llm_call" not in report  # scripted LLM makes no llm_call spans
    assert "doctor → agent (span" in report
