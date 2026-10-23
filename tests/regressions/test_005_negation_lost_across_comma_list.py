"""A negated list of foods ("avoid onion, garlic ...") was read as recommending garlic.

Symptom: COACH-01 (critical) failed on COACH-007 (Jain, diabetic, peanut allergy) with
"has garlic (onion_garlic)" and "has peanut (peanut)".
Where in the trace: COACH-007 -> coach -> llm_call (span 9ff7c67b). The items said
"Avoid onion, garlic and root vegetables." and "do not add onion, garlic or peanuts."
Root cause: the negation check treated every comma as the end of a clause, so "avoid"
covered "onion" but not "garlic" after the comma.
Fix: for food lists, only sentence punctuation or "but" ends a negation. Commas still end
a clause for Ome's claim checks, where "No worries, your LDL is high" must be caught.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.coach_evals import diet_constraints_respected, foods_in
from foxo_evals.evals.text import find_claims
from foxo_evals.system.schemas import CoachPlan

CASE = next(c for c in load_dataset("coach") if c.case_id == "COACH-007")


def plan(item: str) -> CoachPlan:
    return CoachPlan(
        focus_areas=["HbA1c"], diet_recommendations=[{"item": item, "rationale": "fibre", "linked_biomarker": "hba1c"}],
        activity="Walk daily.", sleep="7 hours", retest_in_weeks=12,
    )


def test_negated_comma_lists_are_not_recommendations():
    items = [
        "1/2 plate non-root seasonal sabzi, 1 katori toor dal and 2 jowar rotis. Avoid onion, garlic and root vegetables.",
        "30 g roasted chana or moong sprouts chaat with lemon; do not add onion, garlic or peanuts.",
    ]
    for item in items:
        assert diet_constraints_respected(CASE, plan(item), []).passed, item


def test_a_new_sentence_ends_the_negation():
    assert sorted(food for food, _ in foods_in("Avoid fried snacks. Enjoy peanut chikki.")) == ["chikki", "peanut"]
    assert not diet_constraints_respected(CASE, plan("Dal with garlic tadka"), []).passed


def test_commas_still_end_a_clause_for_ome_claims():
    assert find_claims("No worries, your LDL is high.", ["is high"]) == ["is high"]
