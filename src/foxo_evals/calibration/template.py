"""Export judged items for a human to label. The judge's own verdict is kept out of the CSV."""

import csv
import json
import random
from pathlib import Path

from foxo_evals.config import CALIBRATION_DIR
from foxo_evals.datasets.loader import load_dataset
from foxo_evals.datasets.schemas import CASE_TYPES
from foxo_evals.evals.judge import judge_prompt
from foxo_evals.evals.registry import REGISTRY, EvalResult
from foxo_evals.system.schemas import CoachPlan, DoctorOutput, JourneyResult, LabResult, OmeResponse

LABELS_FILE = "labels_template.csv"
VERDICTS_FILE = "judge_verdicts.json"
COLUMNS = ["item_id", "eval_id", "case_id", "rubric", "what_the_judge_read", "human_label", "notes"]
OUTPUT_TYPES = {"ome": OmeResponse, "mood": LabResult, "doctor": DoctorOutput, "coach": CoachPlan, "e2e": JourneyResult}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def judge_items(run_dir: Path) -> list[dict]:
    """Every judged (eval, case) in a run, with the exact text the judge read."""
    results = [EvalResult.model_validate_json(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    outputs = {row["case_id"]: row for row in map(json.loads, (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines())}
    cases = {case.case_id: case for name in CASE_TYPES for case in load_dataset(name)}
    items = []
    for result in results:
        spec = REGISTRY.get(result.eval_id)
        row = outputs.get(result.case_id)
        if spec is None or spec.kind != "judge" or not row or row["output"] is None:
            continue
        if result.details.startswith("judge call failed"):
            continue
        output = OUTPUT_TYPES[row["dataset"]].model_validate(row["output"])
        criteria = judge_prompt(spec.judge).criteria
        items.append({
            "item_id": f"{result.eval_id}:{result.case_id}",
            "eval_id": result.eval_id,
            "case_id": result.case_id,
            "rubric": "\n".join(f"{cid}: {text}" for cid, text in criteria.items()),
            "what_the_judge_read": spec.subject(cases[result.case_id], output),
            "judge_passed": result.passed,
            "judge_details": result.details,
        })
    return items


def pick(items: list[dict], size: int, seed: int) -> list[dict]:
    """Every judge failure first (they matter most), then passes spread evenly across evals.

    A plain random sample would be almost all passes, and agreement on those says
    little about whether the judge catches real problems.
    """
    rng = random.Random(seed)
    chosen = [item for item in items if not item["judge_passed"]][:size]
    passes: dict[str, list[dict]] = {}
    for item in items:
        if item["judge_passed"]:
            passes.setdefault(item["eval_id"], []).append(item)
    for group in passes.values():
        rng.shuffle(group)
    while len(chosen) < size and any(passes.values()):
        for group in passes.values():
            if group and len(chosen) < size:
                chosen.append(group.pop())
    return sorted(chosen, key=lambda item: item["item_id"])


def export_template(run_dir: Path, out_dir: Path = CALIBRATION_DIR, size: int = 30, seed: int = 7,
                    force: bool = False) -> Path:
    labels_path = out_dir / LABELS_FILE
    if labels_path.exists() and not force and any(row["human_label"].strip() for row in read_rows(labels_path)):
        raise FileExistsError(f"{labels_path} already has human labels. Use --force to overwrite them.")
    items = pick(judge_items(run_dir), size, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({**item, "human_label": "", "notes": ""} for item in items)
    verdicts = {
        "run": run_dir.name,
        "items": {i["item_id"]: {"judge_passed": i["judge_passed"], "judge_details": i["judge_details"]} for i in items},
    }
    (out_dir / VERDICTS_FILE).write_text(json.dumps(verdicts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return labels_path
