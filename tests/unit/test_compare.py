import json

from foxo_evals.compare import compare_runs, render_regression
from foxo_evals.tracing import Span

SEVERITY = {"DOC-03": "critical", "DOC-02": "major", "OME-08": "minor"}


def write_run(path, run_id: str, prompt: str, passed: dict[tuple[str, str], bool]) -> None:
    path.mkdir()
    evals = {}
    for eval_id, severity in SEVERITY.items():
        mine = [ok for (e, _), ok in passed.items() if e == eval_id]
        evals[eval_id] = {"severity": severity, "pass_rate": sum(mine) / len(mine)}
    summary = {"run_id": run_id, "prompts": {"doctor": {"version": prompt}}, "models": {"agent": "m"},
               "dataset": {"version": "1.2.0"}, "evals": evals}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [{"eval_id": e, "case_id": c, "passed": ok, "details": "" if ok else "missed it", "span_id": "s1"}
            for (e, c), ok in passed.items()]
    (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    span = Span(trace_id="DOC-001", span_id="s1", parent_span_id=None, agent="doctor", step="llm_call")
    (path / "traces.jsonl").write_text(span.model_dump_json() + "\n", encoding="utf-8")


def runs(tmp_path, before: dict, after: dict):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "a", "v1", before)
    write_run(b, "b", "v2", after)
    return a, b


def all_pass(n: int = 20) -> dict:
    return {(e, f"C{i}"): True for e in SEVERITY for i in range(n)}


def test_identical_runs_do_not_block(tmp_path):
    a, b = runs(tmp_path, all_pass(), all_pass())
    assert compare_runs(a, b)["blocking"] == []


def test_a_single_critical_flip_blocks(tmp_path):
    after = {**all_pass(), ("DOC-03", "C0"): False}
    a, b = runs(tmp_path, all_pass(), after)
    comparison = compare_runs(a, b)
    assert comparison["blocking"] == ["critical regression: DOC-03 now fails on C0"]
    report = render_regression(comparison, b)
    assert "## Verdict: BLOCK" in report and "doctor → llm_call (span `s1`)" in report
    assert "doctor v1 -> v2" in report


def test_major_drop_blocks_only_beyond_the_threshold(tmp_path):
    small = {**all_pass(), ("DOC-02", "C0"): False}  # 100% -> 95%: within 5 points
    a, b = runs(tmp_path, all_pass(), small)
    assert compare_runs(a, b)["blocking"] == []

    big = {**small, ("DOC-02", "C1"): False}  # 100% -> 90%
    (tmp_path / "second").mkdir()
    a2, b2 = runs(tmp_path / "second", all_pass(), big)
    assert compare_runs(a2, b2)["blocking"] == ["major regression: DOC-02 pass rate 100% -> 90%"]


def test_minor_flips_never_block(tmp_path):
    after = {**all_pass(), **{("OME-08", f"C{i}"): False for i in range(10)}}
    a, b = runs(tmp_path, all_pass(), after)
    comparison = compare_runs(a, b)
    assert comparison["blocking"] == [] and len(comparison["flipped"]) == 10
