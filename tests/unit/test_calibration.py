import csv
import json

import pytest

from foxo_evals.calibration.agreement import calibrate, cohen_kappa
from foxo_evals.calibration.template import LABELS_FILE, export_template, pick, read_rows
from foxo_evals.datasets.loader import load_dataset
from foxo_evals.runner import REGISTRY  # noqa: F401 (registers the judge evals)


def test_kappa_known_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # 3 of 4 agree; chance agreement is 0.5, so kappa = (0.75 - 0.5) / 0.5
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)
    assert cohen_kappa([True, True], [True, True]) is None


def test_pick_takes_every_failure_then_spreads_passes():
    items = [{"item_id": f"A:{i}", "eval_id": "A", "judge_passed": True} for i in range(10)]
    items += [{"item_id": f"B:{i}", "eval_id": "B", "judge_passed": True} for i in range(10)]
    items += [{"item_id": "B:fail", "eval_id": "B", "judge_passed": False}]
    chosen = pick(items, size=5, seed=7)
    assert len(chosen) == 5
    assert any(i["item_id"] == "B:fail" for i in chosen)
    assert {i["eval_id"] for i in chosen} == {"A", "B"}
    assert chosen == pick(items, size=5, seed=7)


@pytest.fixture
def fake_run(tmp_path):
    """A tiny run folder with two judged Ome cases."""
    run = tmp_path / "run"
    run.mkdir()
    cases = load_dataset("ome")[:2]
    output = {"intent": "support", "urgency": "routine", "reply_text": "Happy to help.", "handoff_to": "none",
              "escalated": False, "escalation_reason": None, "cited_policy_ids": []}
    (run / "outputs.jsonl").write_text(
        "".join(json.dumps({"dataset": "ome", "case_id": c.case_id, "output": output, "error": None}) + "\n" for c in cases),
        encoding="utf-8",
    )
    results = [
        {"eval_id": "OME-08", "case_id": cases[0].case_id, "passed": True, "details": "all criteria met"},
        {"eval_id": "OME-08", "case_id": cases[1].case_id, "passed": False, "details": "clear: too vague"},
    ]
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    return run


def test_template_hides_the_judge_verdict(fake_run, tmp_path):
    out = tmp_path / "calibration"
    path = export_template(fake_run, out_dir=out)
    rows = read_rows(path)
    assert len(rows) == 2
    assert all(row["human_label"] == "" for row in rows)
    assert "judge_passed" not in rows[0] and "Ome's reply" in rows[0]["what_the_judge_read"]


def test_template_refuses_to_overwrite_human_labels(fake_run, tmp_path):
    out = tmp_path / "calibration"
    path = export_template(fake_run, out_dir=out)
    path.write_text(path.read_text(encoding="utf-8-sig").replace(",,\n", ",pass,\n", 1), encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_template(fake_run, out_dir=out)


def test_no_labels_means_not_yet_calibrated(fake_run, tmp_path):
    out = tmp_path / "calibration"
    export_template(fake_run, out_dir=out)
    report = calibrate(out, tmp_path / "calibration.md")
    assert "Not yet calibrated" in report and "Agreement" not in report


def label(out, labels: list[str]) -> None:
    """Fill in human_label the way a person would in a spreadsheet."""
    path = out / LABELS_FILE
    rows = read_rows(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, "human_label": value} for row, value in zip(rows, labels))


def test_agreement_and_kappa_from_labels(fake_run, tmp_path):
    out = tmp_path / "calibration"
    export_template(fake_run, out_dir=out)
    label(out, ["fail", "fail"])  # judge said pass, fail
    report = calibrate(out, tmp_path / "calibration.md")
    assert "Agreement: **50%** (1/2)" in report
    assert "## Disagreements" in report and "human fail, judge pass" in report


def test_unrecognised_labels_are_rejected(fake_run, tmp_path):
    out = tmp_path / "calibration"
    export_template(fake_run, out_dir=out)
    label(out, ["maybe", "pass"])
    with pytest.raises(ValueError, match="maybe"):
        calibrate(out, tmp_path / "calibration.md")
