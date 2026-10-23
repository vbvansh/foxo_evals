"""Compare two runs and decide whether B regressed against A. This is the CI gate.

Blocks on any critical eval that flipped from pass to fail on a case, or on a major
eval whose pass rate dropped by more than `major_drop`.
"""

import json
from pathlib import Path

from foxo_evals.evals.registry import EvalResult
from foxo_evals.tracing import load_spans

MAJOR_DROP = 0.05


def load_results(run_dir: Path) -> dict[tuple[str, str], EvalResult]:
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {(r.eval_id, r.case_id): r for r in map(EvalResult.model_validate_json, lines)}


def compare_runs(a_dir: Path, b_dir: Path, major_drop: float = MAJOR_DROP) -> dict:
    a_summary = json.loads((a_dir / "summary.json").read_text(encoding="utf-8"))
    b_summary = json.loads((b_dir / "summary.json").read_text(encoding="utf-8"))
    a_results, b_results = load_results(a_dir), load_results(b_dir)

    flipped = [b_results[key] for key, a in a_results.items() if a.passed and key in b_results and not b_results[key].passed]
    fixed = [b_results[key] for key, a in a_results.items() if not a.passed and key in b_results and b_results[key].passed]
    new_failures = [r for key, r in b_results.items() if key not in a_results and not r.passed]

    deltas = {}
    for eval_id, b in b_summary["evals"].items():
        a = a_summary["evals"].get(eval_id)
        if a:
            deltas[eval_id] = {"severity": b["severity"], "before": a["pass_rate"], "after": b["pass_rate"],
                               "delta": round(b["pass_rate"] - a["pass_rate"], 4)}

    reasons = []
    for r in flipped:
        if b_summary["evals"][r.eval_id]["severity"] == "critical":
            reasons.append(f"critical regression: {r.eval_id} now fails on {r.case_id}")
    for eval_id, d in deltas.items():
        if d["severity"] == "major" and d["delta"] < -major_drop:
            reasons.append(f"major regression: {eval_id} pass rate {d['before']:.0%} -> {d['after']:.0%}")

    return {
        "a": a_summary, "b": b_summary, "flipped": flipped, "fixed": fixed, "new_failures": new_failures,
        "deltas": deltas, "blocking": reasons,
    }


def changed_prompts(a: dict, b: dict) -> list[str]:
    return [
        f"{agent} {a['prompts'][agent]['version']} -> {p['version']}"
        for agent, p in b["prompts"].items() if a["prompts"].get(agent, {}).get("version") != p["version"]
    ]


def render_regression(comparison: dict, b_dir: Path) -> str:
    a, b = comparison["a"], comparison["b"]
    spans = {s.span_id: s for s in load_spans(b_dir / "traces.jsonl")}
    severity = {eval_id: e["severity"] for eval_id, e in b["evals"].items()}
    blocking = comparison["blocking"]
    lines = [
        f"# Regression check: {a['run_id']} -> {b['run_id']}", "",
        f"- Prompt changes: {', '.join(changed_prompts(a, b)) or 'none'}",
        f"- Models: {a['models']['agent']} -> {b['models']['agent']}; dataset {a['dataset']['version']} -> {b['dataset']['version']}",
        "", f"## Verdict: {'BLOCK' if blocking else 'OK'}", "",
    ]
    lines += [f"- {reason}" for reason in blocking] or ["No critical regressions and no major pass-rate drop."]

    lines += ["", "## Evals that flipped from pass to fail", ""]
    for level in ("critical", "major", "minor"):
        group = [r for r in comparison["flipped"] if severity[r.eval_id] == level]
        for r in group:
            span = spans.get(r.span_id or "")
            where = f"{span.agent} → {span.step} (span `{span.span_id}`)" if span else "no span"
            lines.append(f"- **{r.eval_id}** ({level}) `{r.case_id}`: {r.details}  \n  Trace: {where}")
    if not comparison["flipped"]:
        lines.append("None.")

    lines += ["", "## Pass-rate changes", "", "| Eval | Severity | Before | After | Change |", "|---|---|---|---|---|"]
    for eval_id, d in comparison["deltas"].items():
        if d["delta"]:
            lines.append(f"| {eval_id} | {d['severity']} | {d['before']:.0%} | {d['after']:.0%} | {d['delta']:+.0%} |")

    if comparison["fixed"]:
        lines += ["", "## Fixed (fail -> pass)", ""]
        lines += [f"- {r.eval_id} `{r.case_id}`" for r in comparison["fixed"]]
    if comparison["new_failures"]:
        lines += ["", "## Failures on cases that are new in this run", ""]
        lines += [f"- {r.eval_id} `{r.case_id}`: {r.details}" for r in comparison["new_failures"]]
    lines += ["", f"Inspect a trace with `make trace RUN=runs/{b['run_id']} TRACE=<case_id>`.", ""]
    return "\n".join(lines)
