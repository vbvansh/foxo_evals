"""Renders report.md: the gate, a headline table, every failure with a link into the trace."""

from foxo_evals.evals.registry import EvalResult
from foxo_evals.tracing import Span

INTENTS = ["support", "lab_booking", "report_question", "symptom_concern", "prescription_request", "other"]


def trace_link(result: EvalResult, spans_by_id: dict[str, Span]) -> str:
    span = spans_by_id.get(result.span_id or "")
    if span is None:
        return "no span"
    return f"{span.agent} → {span.step} (span `{span.span_id}`)"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_report(summary: dict, results: list[EvalResult], spans: list[Span]) -> str:
    spans_by_id = {s.span_id: s for s in spans}
    gate = summary["gate"]
    prompts = ", ".join(f"{agent} {p['version']}" for agent, p in summary["prompts"].items())
    lines = [
        f"# Eval report: {summary['run_id']}",
        "",
        f"- Mode: {summary['mode']} · commit `{summary['git_commit']}` · {summary['timestamp']}",
        f"- Models: agents `{summary['models']['agent']}`, judge `{summary['models']['judge']}`",
        f"- Prompts: {prompts}",
        f"- Dataset version: {summary['dataset']['version']} "
        f"({', '.join(f'{k} {v}' for k, v in summary['cases'].items())} cases)",
        "",
        f"## Gate: {'PASS' if gate['passed'] else 'FAIL'}",
        "",
    ]
    lines += [f"- {reason}" for reason in gate["reasons"]] or ["All critical evals passed and every major eval met its threshold."]

    lines += ["", "## Headline", "", "| Severity | Evals | Checks | Passed | Pass rate |", "|---|---|---|---|---|"]
    for severity, s in summary["severity"].items():
        lines.append(f"| {severity} | {s['evals']} | {s['checks']} | {s['passed']} | {pct(s['pass_rate'])} |")

    lines += ["", "## Every eval", "", "| ID | Agent | Kind | Severity | Passed | Rate | Status | Check |",
              "|---|---|---|---|---|---|---|---|"]
    for eval_id, s in summary["evals"].items():
        threshold = f" (≥{s['threshold']:.0%})" if s["threshold"] else ""
        lines.append(
            f"| {eval_id} | {s['agent']} | {s['kind']} | {s['severity']} | {s['passed']}/{s['total']} "
            f"| {pct(s['pass_rate'])}{threshold} | {s['status']} | {s['description']} |"
        )

    lines += ["", "## Failures", ""]
    failures = [r for r in results if not r.passed]
    if not failures:
        lines.append("None.")
    for severity in ("critical", "major", "minor"):
        group = [r for r in failures if summary["evals"][r.eval_id]["severity"] == severity]
        if group:
            lines += [f"### {severity.capitalize()} ({len(group)})", ""]
            lines += [
                f"- **{r.eval_id}** `{r.case_id}`: {r.details}  \n  Trace: {trace_link(r, spans_by_id)}"
                for r in group
            ]
            lines.append("")

    routing = summary["ome_routing"]
    lines += ["", "## Ome routing", "",
              f"Escalation recall: {pct(routing['escalation_recall'])} · "
              f"false escalation rate: {pct(routing['false_escalation_rate'])}", "",
              "Intent confusion matrix (rows: gold, columns: predicted):", "",
              "| gold \\ predicted | " + " | ".join(INTENTS + ["error"]) + " |",
              "|---" * (len(INTENTS) + 2) + "|"]
    for gold in INTENTS:
        row = routing["confusion"].get(gold)
        if row:
            lines.append(f"| {gold} | " + " | ".join(str(row.get(p, "")) for p in INTENTS + ["error"]) + " |")

    latency = summary["journey_latency_ms"]
    lines += ["", "## Cost and latency", "",
              f"- Total model cost for this run: ${summary['cost_usd']:.4f} "
              f"(agents ${summary['cost_usd'] - summary['judge_cost_usd']:.4f}, judges ${summary['judge_cost_usd']:.4f})",
              f"- End-to-end journey model time: p50 {fmt_seconds(latency['p50'])}, p95 {fmt_seconds(latency['p95'])}",
              "", f"Inspect any case with `make trace RUN=runs/{summary['run_id']} TRACE=<case_id>`.", ""]
    return "\n".join(lines)


def fmt_seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f} s"
