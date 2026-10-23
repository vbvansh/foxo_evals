"""Command line entry point: python -m foxo_evals <command>."""

import argparse
import sys
from pathlib import Path

from foxo_evals.tracing import format_trace, load_spans


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

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
