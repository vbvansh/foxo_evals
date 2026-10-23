"""Deterministic checks for Mood, the lab-report agent."""

from pydantic import ValidationError

from foxo_evals.datasets.schemas import MoodCase
from foxo_evals.evals.registry import Verdict, eval_check, fail, ok, verdict_from
from foxo_evals.evals.text import SPECS, close, numbers_in
from foxo_evals.system.normalise import describe, parse_value
from foxo_evals.system.schemas import LabResult
from foxo_evals.tracing import Span


@eval_check(
    id="MOOD-01", agent="mood", kind="deterministic", severity="major",
    description="Extraction precision and recall against gold biomarkers (name, value, unit)",
)
def extraction_precision_recall(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    gold = {g.name: g for g in case.gold.biomarkers}
    found = {b.name: b for b in output.biomarkers}
    correct = {
        name for name, g in gold.items()
        if name in found and found[name].unit == g.unit and close(found[name].value, g.value)
    }
    precision = len(correct) / len(found) if found else 1.0
    recall = len(correct) / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    details = f"precision {precision:.2f}, recall {recall:.2f}"
    if missing := sorted(set(gold) - correct):
        details += f"; missing or wrong: {missing}"
    if extra := sorted(set(found) - set(gold)):
        details += f"; not in gold: {extra}"
    if precision == recall == 1.0:
        return ok(details, score=1.0)
    return fail(details, score=round(f1, 3))


@eval_check(
    id="MOOD-02", agent="mood", kind="deterministic", severity="critical",
    description="Every extracted value appears as a number in the raw report",
)
def values_grounded(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    printed = set(numbers_in(case.report.raw_text))
    ungrounded = []
    for b in output.biomarkers:
        parsed = parse_value(b.source.value_as_printed)
        if parsed is None or parsed[0] not in printed:
            ungrounded.append(describe(b.source))
    return verdict_from([f"values not in the report: {ungrounded}"] if ungrounded else [])


@eval_check(
    id="MOOD-03", agent="mood", kind="deterministic", severity="critical",
    description="Converted values match gold within tolerance",
    step="normalise_units",
)
def converted_values_correct(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    gold = {g.name: g for g in case.gold.biomarkers}
    wrong = []
    for b in output.biomarkers:
        g = gold.get(b.name)
        if g and (not close(b.value, g.value) or b.comparator != g.comparator):
            wrong.append(
                f"{b.name}: read '{describe(b.source)}' -> {b.comparator}{b.value} {b.unit}, "
                f"expected {g.comparator}{g.value} {g.unit}"
            )
    return verdict_from(wrong)


@eval_check(
    id="MOOD-04a", agent="mood", kind="deterministic", severity="critical",
    description="Every critical value is flagged critical (critical recall must be 100%)",
    applies=lambda case: any(g.flag.startswith("critical") for g in case.gold.biomarkers),
    step="normalise_units",
)
def critical_flags_recalled(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    found = {b.name: b.flag for b in output.biomarkers}
    missed = [
        f"{g.name}: expected {g.flag}, got {found.get(g.name, 'not extracted')}"
        for g in case.gold.biomarkers
        if g.flag.startswith("critical") and found.get(g.name) != g.flag
    ]
    return verdict_from(missed)


@eval_check(
    id="MOOD-04b", agent="mood", kind="deterministic", severity="major",
    description="All flags match gold for the biomarkers that were extracted",
    step="normalise_units",
)
def flags_correct(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    gold = {g.name: g.flag for g in case.gold.biomarkers}
    wrong = [
        f"{b.name}: {b.flag}, expected {gold[b.name]}"
        for b in output.biomarkers
        if b.name in gold and gold[b.name] != b.flag
    ]
    return verdict_from(wrong)


@eval_check(
    id="MOOD-05", agent="mood", kind="deterministic", severity="critical",
    description="Unreadable or ambiguous values go to unparsed_items and are never invented",
    applies=lambda case: bool(case.gold.unparsed),
)
def unreadable_not_guessed(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    found = {b.name: b for b in output.biomarkers}
    problems = []
    for name in case.gold.unparsed:
        if name in found:
            problems.append(f"invented a value for {name}: '{describe(found[name].source)}'")
        if not any(name in item.lower() for item in output.unparsed_items):
            problems.append(f"{name} not listed in unparsed_items")
    return verdict_from(problems)


@eval_check(
    id="MOOD-06", agent="mood", kind="deterministic", severity="major",
    description="Sample quality issues are detected (and none are invented)",
    step="normalise_units",
)
def quality_issues_detected(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    expected, got = set(case.gold.quality_issues), set(output.quality_issues)
    if expected == got:
        return ok()
    return fail(f"expected {sorted(expected)}, got {sorted(got)}")


@eval_check(
    id="MOOD-07", agent="mood", kind="deterministic", severity="critical",
    description="Output is schema-valid and consistent with the reference data",
    step="normalise_units",
)
def output_schema_valid(case: MoodCase, output: LabResult, trace: list[Span]) -> Verdict:
    try:
        LabResult.model_validate(output.model_dump())
    except ValidationError as error:
        return verdict_from([f"schema: {error}"])
    problems = []
    for b in output.biomarkers:
        spec = SPECS.get(b.name)
        if spec is None:
            problems.append(f"unknown biomarker {b.name}")
        elif b.unit != spec.canonical_unit or (b.ref_low, b.ref_high) != spec.ref[case.report.member_sex]:
            problems.append(f"{b.name}: unit or reference range does not match the reference data")
    return verdict_from(problems)
