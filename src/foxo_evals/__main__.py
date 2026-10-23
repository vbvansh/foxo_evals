"""Command line entry point: python -m foxo_evals <command>."""

import argparse
import sys
from pathlib import Path

from foxo_evals.datasets.loader import check_data, read_manifest, update_manifest
from foxo_evals.tracing import format_trace, load_spans


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

    trace = commands.add_parser("trace", help="print the span tree for one case")
    trace.add_argument("run_dir", type=Path)
    trace.add_argument("trace_id")
    trace.set_defaults(handler=trace_command)

    check = commands.add_parser("check-data", help="validate datasets and compare hashes with the manifest")
    check.set_defaults(handler=check_data_command)

    bump = commands.add_parser("update-manifest", help="record new hashes after an intended data change")
    bump.add_argument("--version", required=True)
    bump.add_argument("--note", required=True)
    bump.set_defaults(handler=update_manifest_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
