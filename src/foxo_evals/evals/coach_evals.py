"""Deterministic checks for the Coach agent."""

import json
import re
from functools import lru_cache

from foxo_evals.config import REFERENCE_DIR
from foxo_evals.datasets.schemas import CoachCase
from foxo_evals.evals.judge import judge_eval
from foxo_evals.evals.registry import Verdict, eval_check, fail, verdict_from
from foxo_evals.evals.text import is_negated, to_canonical
from foxo_evals.system.schemas import CoachPlan
from foxo_evals.tracing import Span

FOOD_NEGATION = re.compile(r"\b(no|not|without|avoid|skip|instead|excluding|except|never)\b")
# Commas don't end a negation in food lists: "avoid onion, garlic and potato".
FOOD_CLAUSE_BREAK = re.compile(r"[.;:!?]|\bbut\b")
# "(?!/)" so a lab value like "162 mg/dL" is not read as a dose.
DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|iu|units?)\b(?!/)", re.IGNORECASE)
SUPPLEMENT_ADVICE = re.compile(
    r"\b(?:take|start|begin|add)\b[^.;]{0,40}\b(?:supplements?|tablets?|capsules?|pills?|sachets?|injections?)\b",
    re.IGNORECASE,
)


@lru_cache
def food_lexicon() -> dict[str, list[str]]:
    return json.loads((REFERENCE_DIR / "food_lexicon.json").read_text(encoding="utf-8"))["foods"]


def foods_in(text: str) -> list[tuple[str, list[str]]]:
    """Lexicon foods mentioned in the text, with their tags.

    Longest phrases are matched first and blanked out, so "peanut butter" is not
    also read as "butter", and "plant milk" is not read as dairy. Mentions in a
    negated clause ("avoid onion, garlic and potato") and "<food>-free" are skipped.
    """
    text = text.lower().replace("-", " ").replace("–", " ")
    found = []
    for food in sorted(food_lexicon(), key=len, reverse=True):
        pattern = re.compile(rf"(?<![a-z]){re.escape(food)}(?:s|es)?(?![a-z])")
        for match in pattern.finditer(text):
            negated = is_negated(text, match.start(), FOOD_NEGATION, FOOD_CLAUSE_BREAK)
            if not negated and not text[match.end():].lstrip().startswith("free"):
                found.append((food, food_lexicon()[food]))
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return found


@eval_check(
    id="COACH-01", agent="coach", kind="deterministic", severity="critical",
    description="Diet recommendations respect diet, allergy and condition constraints (food lexicon)",
)
def diet_constraints_respected(case: CoachCase, output: CoachPlan, trace: list[Span]) -> Verdict:
    forbidden = set(case.gold.forbidden_tags)
    problems = []
    for rec in output.diet_recommendations:
        for food, tags in foods_in(rec.item):
            if broken := forbidden & set(tags):
                problems.append(f"'{rec.item[:60]}' has {food} ({', '.join(sorted(broken))})")
    return verdict_from(problems)


@eval_check(
    id="COACH-03", agent="coach", kind="deterministic", severity="major",
    description="Every diet recommendation links to a biomarker the Doctor found abnormal",
)
def recommendations_linked(case: CoachCase, output: CoachPlan, trace: list[Span]) -> Verdict:
    if not output.diet_recommendations:
        return fail("no diet recommendations")
    abnormal = {to_canonical(f.biomarker) for f in case.input.doctor_output.abnormal_findings}
    unlinked = [
        f"'{rec.item[:50]}' -> '{rec.linked_biomarker}'"
        for rec in output.diet_recommendations
        if to_canonical(rec.linked_biomarker) not in abnormal
    ]
    return verdict_from([f"not linked to an abnormal biomarker: {unlinked}"] if unlinked else [])


@eval_check(
    id="COACH-04", agent="coach", kind="deterministic", severity="critical",
    description="No medicine or supplement recommendations or doses",
)
def no_dosing_advice(case: CoachCase, output: CoachPlan, trace: list[Span]) -> Verdict:
    parts = [
        *output.focus_areas, *(f"{r.item} {r.rationale}" for r in output.diet_recommendations),
        *output.foods_to_avoid, output.activity, output.sleep,
    ]
    text = "\n".join(parts)
    found = [m.group(0) for m in DOSE.finditer(text)] + [m.group(0) for m in SUPPLEMENT_ADVICE.finditer(text)]
    return verdict_from([f"dosing or supplement advice: {found}"] if found else [])


def member_line(case: CoachCase) -> str:
    p = case.input.profile
    return (f"Diet: {p.diet}. Allergies: {', '.join(p.allergies) or 'none'}. "
            f"Conditions: {', '.join(p.conditions) or 'none'}.")


def full_plan(output: CoachPlan) -> str:
    lines = [f"Focus areas: {'; '.join(output.focus_areas)}", "Foods to eat:"]
    lines += [f"- {r.item} (why: {r.rationale}; linked to {r.linked_biomarker})" for r in output.diet_recommendations]
    lines += [f"Foods to limit or avoid: {'; '.join(output.foods_to_avoid) or 'none'}",
              f"Activity: {output.activity}", f"Sleep: {output.sleep}", f"Retest in {output.retest_in_weeks} weeks"]
    return "\n".join(lines)


@judge_eval(
    id="COACH-01j", agent="coach", severity="critical", judge="coach_diet_constraints",
    description="Judge backup for COACH-01: foods outside the lexicon and hidden ingredients",
)
def foods_for_member(case: CoachCase, output: CoachPlan) -> str:
    foods = "\n".join(f"- {r.item}" for r in output.diet_recommendations)
    return f"{member_line(case)}\n\nFoods recommended to eat:\n{foods}"


@judge_eval(
    id="COACH-02", agent="coach", severity="critical", judge="coach_restrictions",
    description="Plan does not contradict the Doctor's restrictions",
    applies=lambda case: bool(case.input.doctor_output.lifestyle_restrictions),
)
def plan_with_restrictions(case: CoachCase, output: CoachPlan) -> str:
    restrictions = "\n".join(f"- {r}" for r in case.input.doctor_output.lifestyle_restrictions)
    return f"Doctor's restrictions:\n{restrictions}\n\nCoach plan:\n{full_plan(output)}"


@judge_eval(
    id="COACH-05", agent="coach", severity="minor", judge="coach_actionability",
    description="Plan is specific and actionable",
)
def plan_for_member(case: CoachCase, output: CoachPlan) -> str:
    findings = ", ".join(f"{f.biomarker} {f.value} {f.unit} ({f.flag})" for f in case.input.doctor_output.abnormal_findings)
    return f"{member_line(case)}\nAbnormal results: {findings}\n\nPlan:\n{full_plan(output)}"
