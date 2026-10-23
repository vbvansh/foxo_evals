"""Command line entry point: python -m foxo_evals <command>."""

import argparse
import sys
from pathlib import Path

from foxo_evals.calibration.agreement import calibrate
from foxo_evals.calibration.template import export_template
from foxo_evals.compare import MAJOR_DROP, compare_runs, render_regression
from foxo_evals.config import MODE, RUNS_DIR
from foxo_evals.datasets.loader import check_data, read_manifest, update_manifest
from foxo_evals.llm.client import prune_recordings
from foxo_evals.runner import parse_prompt_versions, run_evals
from foxo_evals.tracing import format_trace, load_spans


def run_command(args: argparse.Namespace) -> int:
    datasets = args.datasets.split(",") if args.datasets else None
    summary = run_evals(args.out, mode=args.mode, prompt_versions=parse_prompt_versions(args.prompts), datasets=datasets)
    gate = summary["gate"]
    print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'}   report: {args.out / 'report.md'}")
    for reason in gate["reasons"]:
        print(f"  {reason}")
    return 0 if gate["passed"] else 1


def compare_command(args: argparse.Namespace) -> int:
    comparison = compare_runs(args.a, args.b, args.major_drop)
    text = render_regression(comparison, args.b)
    (args.b / "regression.md").write_text(text, encoding="utf-8")
    print(text)
    return 1 if comparison["blocking"] else 0


def prune_command(args: argparse.Namespace) -> int:
    removed = prune_recordings([run / "traces.jsonl" for run in args.runs])
    print(f"Removed {len(removed)} recording(s) that none of these runs used.")
    return 0


def check_data_command(args: argparse.Namespace) -> int:
    problems = check_data()
    if problems:
        print("Dataset check FAILED:")
        print("\n".join(problems))
        return 1
    manifest = read_manifest()
    counts = ", ".join(f"{name} {count}" for name, count in manifest["case_counts"].items())
    print(f"Datasets OK: version {manifest['version']} ({counts})")
    return 0


def update_manifest_command(args: argparse.Namespace) -> int:
    manifest = update_manifest(args.version, args.note)
    print(f"Manifest updated to {manifest['version']}")
    return 0


def calibration_template_command(args: argparse.Namespace) -> int:
    path = export_template(args.run, size=args.size, force=args.force)
    print(f"Wrote {path}. Fill in human_label (pass or fail) for each row, then run: make calibrate")
    return 0


def calibrate_command(args: argparse.Namespace) -> int:
    print(calibrate())
    return 0


def trace_command(args: argparse.Namespace) -> int:
    spans = [s for s in load_spans(args.run_dir / "traces.jsonl") if s.trace_id == args.trace_id]
    if not spans:
        print(f"No trace '{args.trace_id}' in {args.run_dir}")
        return 1
    print(format_trace(spans))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="foxo_evals")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the agents over the datasets and apply every eval")
    run.add_argument("--mode", choices=["live", "record", "replay"], default=MODE)
    run.add_argument("--prompts", default="v1", help="'v1', or per agent: 'v1,doctor=v2'")
    run.add_argument("--out", type=Path, default=RUNS_DIR / "latest")
    run.add_argument("--datasets", help="comma-separated subset, e.g. 'doctor,e2e' (default: all)")
    run.set_defaults(handler=run_command)

    compare = commands.add_parser("compare", help="compare run B against run A; exit 1 on a regression")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--major-drop", type=float, default=MAJOR_DROP)
    compare.set_defaults(handler=compare_command)

    prune = commands.add_parser("prune-recordings", help="delete recordings none of the given runs used")
    prune.add_argument("runs", type=Path, nargs="+")
    prune.set_defaults(handler=prune_command)

    trace = commands.add_parser("trace", help="print the span tree for one case")
    trace.add_argument("run_dir", type=Path)
    trace.add_argument("trace_id")
    trace.set_defaults(handler=trace_command)

    check = commands.add_parser("check-data", help="validate datasets and compare hashes with the manifest")
    check.set_defaults(handler=check_data_command)

    template = commands.add_parser("calibration-template", help="export judged items for human labelling")
    template.add_argument("--run", type=Path, default=RUNS_DIR / "baseline_v1")
    template.add_argument("--size", type=int, default=30)
    template.add_argument("--force", action="store_true", help="overwrite a CSV that already has labels")
    template.set_defaults(handler=calibration_template_command)

    agreement = commands.add_parser("calibrate", help="compare human labels with judge verdicts")
    agreement.set_defaults(handler=calibrate_command)

    bump = commands.add_parser("update-manifest", help="record new hashes after an intended data change")
    bump.add_argument("--version", required=True)
    bump.add_argument("--note", required=True)
    bump.set_defaults(handler=update_manifest_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
