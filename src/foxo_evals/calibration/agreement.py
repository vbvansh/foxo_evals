"""Compare human labels with judge verdicts: agreement, Cohen's kappa and a confusion matrix.

Nothing here invents a number: with no human labels the report says "not yet calibrated".
"""

import json
from pathlib import Path

from foxo_evals.calibration.template import LABELS_FILE, VERDICTS_FILE, read_rows
from foxo_evals.config import CALIBRATION_DIR, JUDGE_MODEL, REPORTS_DIR

PASS_WORDS = {"pass", "p", "yes", "y", "1", "true"}
FAIL_WORDS = {"fail", "f", "no", "n", "0", "false"}


def parse_label(text: str) -> bool | None:
    word = text.strip().lower()
    if word in PASS_WORDS:
        return True
    if word in FAIL_WORDS:
        return False
    return None


def cohen_kappa(human: list[bool], judge: list[bool]) -> float | None:
    """Agreement corrected for chance. None when chance agreement is already 100% (one class only)."""
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge)) / n
    human_pass, judge_pass = sum(human) / n, sum(judge) / n
    expected = human_pass * judge_pass + (1 - human_pass) * (1 - judge_pass)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def calibrate(calibration_dir: Path = CALIBRATION_DIR, report_path: Path = REPORTS_DIR / "calibration.md") -> str:
    rows = read_rows(calibration_dir / LABELS_FILE)
    verdicts = json.loads((calibration_dir / VERDICTS_FILE).read_text(encoding="utf-8"))
    labelled, bad = [], []
    for row in rows:
        if not row["human_label"].strip():
            continue
        human = parse_label(row["human_label"])
        if human is None:
            bad.append(f"  {row['item_id']}: {row['human_label']!r}")
        else:
            labelled.append((row, human, verdicts["items"][row["item_id"]]["judge_passed"]))
    if bad:
        raise ValueError("Unrecognised human_label values (use pass or fail):\n" + "\n".join(bad))

    lines = [
        "# Judge calibration", "",
        f"Judge model `{JUDGE_MODEL}`, judge verdicts from run `{verdicts['run']}`. "
        f"{len(labelled)} of {len(rows)} items have a human label.", "",
    ]
    if not labelled:
        lines.append("**Not yet calibrated.** Fill in `human_label` (pass or fail) in "
                     f"`data/calibration/{LABELS_FILE}` and run `make calibrate`.")
    else:
        human = [h for _, h, _ in labelled]
        judge = [j for _, _, j in labelled]
        agreement = sum(h == j for h, j in zip(human, judge)) / len(labelled)
        kappa = cohen_kappa(human, judge)

        def count(h: bool, j: bool) -> int:
            return sum(1 for pair in zip(human, judge) if pair == (h, j))

        lines += [
            f"- Agreement: **{agreement:.0%}** ({sum(h == j for h, j in zip(human, judge))}/{len(labelled)})",
            f"- Cohen's kappa: **{'n/a (only one class present)' if kappa is None else f'{kappa:.2f}'}**",
            "", "| | judge pass | judge fail |", "|---|---|---|",
            f"| human pass | {count(True, True)} | {count(True, False)} |",
            f"| human fail | {count(False, True)} | {count(False, False)} |",
            "", "## By eval", "", "| Eval | Labelled | Agreement |", "|---|---|---|",
        ]
        for eval_id in sorted({row["eval_id"] for row, _, _ in labelled}):
            mine = [(h, j) for row, h, j in labelled if row["eval_id"] == eval_id]
            lines.append(f"| {eval_id} | {len(mine)} | {sum(h == j for h, j in mine) / len(mine):.0%} |")
        disagreements = [(row, h, j) for row, h, j in labelled if h != j]
        lines += ["", "## Disagreements", ""]
        lines += [
            f"- `{row['item_id']}`: human {'pass' if h else 'fail'}, judge {'pass' if j else 'fail'}. "
            f"Judge said: {verdicts['items'][row['item_id']]['judge_details']}"
            + (f" Human note: {row['notes']}" if row.get("notes") else "")
            for row, h, j in disagreements
        ] or ["None."]
    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text
