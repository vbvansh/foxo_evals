"""Runs every agent over its dataset, applies the evals, and writes the run folder.

runs/<run_id>/
  traces.jsonl    one span per line
  outputs.jsonl   what each agent produced for each case
  results.jsonl   one EvalResult per (eval, case)
  summary.json    versions, pass rates, cost, latency and the gate
  report.md       the readable report
"""

import json
import math
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from foxo_evals.config import AGENT_MODEL, JUDGE_MODEL, ROOT
from foxo_evals.datasets.loader import DatasetError, check_data, load_dataset, read_manifest
from foxo_evals.datasets.schemas import CASE_TYPES, Case
from foxo_evals.evals import coach_evals, doctor_evals, judge, mood_evals, ome_evals, system_evals  # noqa: F401 (registers evals)
from foxo_evals.evals.registry import REGISTRY, EvalResult
from foxo_evals.evals.system_evals import journey_latency_ms
from foxo_evals.llm.client import LLMClient, LLMError, Mode
from foxo_evals.prompts import load_prompt
from foxo_evals.report import render_report
from foxo_evals.system.coach import run_coach
from foxo_evals.system.context import AgentContext
from foxo_evals.system.doctor import run_doctor
from foxo_evals.system.mood import run_mood
from foxo_evals.system.ome import run_ome
from foxo_evals.system.orchestrator import run_member_journey
from foxo_evals.tracing import Tracer

AGENTS = ("ome", "mood", "doctor", "coach")
RUN_AGENT: dict[str, Callable[[Any, AgentContext], BaseModel]] = {
    "ome": lambda case, ctx: run_ome(case.message, ctx),
    "mood": lambda case, ctx: run_mood(case.report, ctx),
    "doctor": lambda case, ctx: run_doctor(case.input, ctx),
    "coach": lambda case, ctx: run_coach(case.input, ctx),
    "e2e": lambda case, ctx: run_member_journey(case.journey, ctx),
}


@dataclass
class CaseRun:
    dataset: str
    case: Case
    output: BaseModel | None
    error: str | None


def parse_prompt_versions(text: str) -> dict[str, str]:
    """'v1' puts every agent on v1; 'v1,doctor=v2' puts the Doctor on v2 and the rest on v1."""
    versions = {agent: "v1" for agent in AGENTS}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" in part:
            agent, version = (s.strip() for s in part.split("=", 1))
            if agent not in versions:
                raise ValueError(f"unknown agent {agent!r}")
            versions[agent] = version
        else:
            versions = {agent: part for agent in versions}
    return versions


def run_cases(datasets: dict[str, list[Case]], ctx: AgentContext) -> list[CaseRun]:
    runs = []
    total = sum(len(cases) for cases in datasets.values())
    for name, cases in datasets.items():
        for case in cases:
            print(f"[{len(runs) + 1}/{total}] {case.case_id}", flush=True)
            output, error = None, None
            with ctx.tracer.trace(case.case_id):
                try:
                    output = RUN_AGENT[name](case, ctx)
                except (LLMError, ValidationError) as e:
                    error = f"{type(e).__name__}: {e}"
            runs.append(CaseRun(name, case, output, error))
    return runs


def evaluate(runs: list[CaseRun], tracer: Tracer) -> list[EvalResult]:
    results = []
    for spec in REGISTRY.values():
        for run in runs:
            if run.dataset != spec.dataset or not spec.applies(run.case):
                continue
            trace = tracer.spans_for(run.case.case_id)
            if run.output is None:
                failed_span = next((s for s in trace if s.status == "error"), None)
                results.append(EvalResult(
                    eval_id=spec.id, case_id=run.case.case_id, passed=False,
                    details=f"agent produced no output: {run.error}", trace_id=run.case.case_id,
                    span_id=failed_span.span_id if failed_span else None,
                ))
                continue
            try:
                if spec.kind == "judge":
                    # Reopen the case's trace so the judge's own LLM call is traced with it.
                    with tracer.trace(run.case.case_id):
                        results.append(spec.run(run.case, run.output, trace))
                else:
                    results.append(spec.run(run.case, run.output, trace))
            except LLMError as error:
                results.append(EvalResult(
                    eval_id=spec.id, case_id=run.case.case_id, passed=False,
                    details=f"judge call failed: {type(error).__name__}: {error}", trace_id=run.case.case_id,
                ))
            except Exception as error:  # a broken check is a failed check, never a crashed run
                results.append(EvalResult(
                    eval_id=spec.id, case_id=run.case.case_id, passed=False,
                    details=f"check crashed: {type(error).__name__}: {error}", trace_id=run.case.case_id,
                ))
    return results


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; fine for a handful of journeys."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)] if ordered else 0.0


def eval_stats(results: list[EvalResult]) -> dict[str, dict]:
    stats = {}
    order = ["ome", "mood", "doctor", "coach", "system"]
    for spec in sorted(REGISTRY.values(), key=lambda s: (order.index(s.agent), s.id)):
        mine = [r for r in results if r.eval_id == spec.id]
        if not mine:
            continue
        passed = sum(r.passed for r in mine)
        rate = passed / len(mine)
        if spec.severity == "critical":
            status = "pass" if passed == len(mine) else "fail"
        elif spec.severity == "major":
            status = "pass" if rate >= spec.threshold else "fail"
        else:
            status = "tracked"
        stats[spec.id] = {
            "agent": spec.agent, "kind": spec.kind, "severity": spec.severity, "description": spec.description,
            "passed": passed, "total": len(mine), "pass_rate": round(rate, 4),
            "threshold": spec.threshold if spec.severity == "major" else None, "status": status,
        }
    return stats


def gate(stats: dict[str, dict]) -> dict:
    """Critical: any failure blocks. Major: blocks below its pass-rate threshold. Minor: never blocks."""
    reasons = []
    for eval_id, s in stats.items():
        if s["severity"] == "critical" and s["status"] == "fail":
            reasons.append(f"{eval_id}: {s['total'] - s['passed']} critical failure(s)")
        if s["severity"] == "major" and s["status"] == "fail":
            reasons.append(f"{eval_id}: pass rate {s['pass_rate']:.0%} is below {s['threshold']:.0%}")
    return {"passed": not reasons, "reasons": reasons}


def severity_stats(stats: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for severity in ("critical", "major", "minor"):
        rows = [s for s in stats.values() if s["severity"] == severity]
        passed, total = sum(s["passed"] for s in rows), sum(s["total"] for s in rows)
        out[severity] = {"evals": len(rows), "checks": total, "passed": passed,
                         "pass_rate": round(passed / total, 4) if total else None}
    return out


def ome_routing(runs: list[CaseRun]) -> dict:
    """Intent confusion matrix and escalation recall / false-escalation rate for Ome."""
    confusion: dict[str, dict[str, int]] = {}
    escalated_when_needed, needed, escalated_when_not, not_needed = 0, 0, 0, 0
    for run in (r for r in runs if r.dataset == "ome"):
        predicted = run.output.intent if run.output else "error"
        row = confusion.setdefault(run.case.gold.intent, {})
        row[predicted] = row.get(predicted, 0) + 1
        escalated = bool(run.output and run.output.escalated)
        if run.case.gold.escalate:
            needed += 1
            escalated_when_needed += escalated
        else:
            not_needed += 1
            escalated_when_not += escalated
    return {
        "confusion": confusion,
        "escalation_recall": round(escalated_when_needed / needed, 4) if needed else None,
        "false_escalation_rate": round(escalated_when_not / not_needed, 4) if not_needed else None,
    }


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return commit.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_evals(out_dir: Path, mode: Mode = "replay", prompt_versions: dict[str, str] | None = None,
              llm: Any = None, judge_llm: Any = None, datasets: list[str] | None = None) -> dict:
    """Run everything and write the run folder. Refuses to run on data that doesn't match the manifest.

    `datasets` limits the run to some datasets, e.g. ["doctor"] while iterating on the Doctor prompt.
    """
    problems = check_data()
    if problems:
        raise DatasetError("Dataset check failed:\n" + "\n".join(problems))
    prompt_versions = prompt_versions or parse_prompt_versions("v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("traces.jsonl", "outputs.jsonl", "results.jsonl", "summary.json", "report.md"):
        (out_dir / name).unlink(missing_ok=True)

    tracer = Tracer(out_dir / "traces.jsonl")
    ctx = AgentContext(llm=llm or LLMClient(AGENT_MODEL, tracer, mode=mode), tracer=tracer,
                       prompt_versions=prompt_versions)
    judge.set_client(judge_llm or LLMClient(JUDGE_MODEL, tracer, mode=mode))
    selected = {name: load_dataset(name) for name in (datasets or CASE_TYPES)}
    runs = run_cases(selected, ctx)
    results = evaluate(runs, tracer)

    stats = eval_stats(results)
    manifest = read_manifest()
    e2e_latencies = [journey_latency_ms(tracer.spans_for(r.case.case_id)) for r in runs if r.dataset == "e2e"]
    llm_calls = [s for s in tracer.spans if s.step == "llm_call"]
    judge_cost = sum(s.cost_usd for s in llm_calls if s.agent.startswith("judges/"))
    judges_used = sorted({spec.judge for spec in REGISTRY.values() if spec.judge})
    summary = {
        "run_id": out_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "git_commit": git_commit(),
        "models": {"agent": AGENT_MODEL, "judge": JUDGE_MODEL},
        "prompts": {agent: {"version": v, "date": str(load_prompt(agent, v).date)} for agent, v in prompt_versions.items()},
        "judge_prompts": {name: judge.JUDGE_PROMPT_VERSION for name in judges_used},
        "dataset": {"version": manifest["version"], "files": manifest["files"]},
        "cases": {name: len(cases) for name, cases in selected.items()},
        "evals": stats,
        "severity": severity_stats(stats),
        "ome_routing": ome_routing(runs),
        "cost_usd": round(sum(s.cost_usd for s in llm_calls), 6),
        "judge_cost_usd": round(judge_cost, 6),
        "journey_latency_ms": {
            "p50": round(statistics.median(e2e_latencies), 1) if e2e_latencies else None,
            "p95": round(percentile(e2e_latencies, 95), 1) if e2e_latencies else None,
        },
        "gate": gate(stats),
    }

    write_jsonl(out_dir / "outputs.jsonl", [
        {"dataset": r.dataset, "case_id": r.case.case_id,
         "output": r.output.model_dump(mode="json") if r.output else None, "error": r.error}
        for r in runs
    ])
    write_jsonl(out_dir / "results.jsonl", [r.model_dump() for r in results])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary, results, tracer.spans), encoding="utf-8")
    return summary
