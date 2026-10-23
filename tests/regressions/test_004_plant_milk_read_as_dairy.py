"""Fortified plant milk in a vegan plan was read as dairy.

Symptom: COACH-01 (critical) failed on COACH-006 (vegan) with "'B12-fortified plant milk
(soy or almond) ...' has milk (dairy)".
Where in the trace: COACH-006 -> coach -> llm_call (span 95e261c1).
Root cause: a gap in the food lexicon. "plant milk" wasn't listed, so its "milk" matched
the dairy entry. (Soy and coconut milk were already listed as not dairy.)
Fix: added "plant milk" and "oat milk" with no tags. Dataset manifest bumped to 1.2.0.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.coach_evals import diet_constraints_respected
from foxo_evals.system.schemas import CoachPlan

CASE = next(c for c in load_dataset("coach") if c.case_id == "COACH-006")


def plan(item: str) -> CoachPlan:
    return CoachPlan(
        focus_areas=["B12"], diet_recommendations=[{"item": item, "rationale": "B12", "linked_biomarker": "vitamin_b12"}],
        activity="Walk daily.", sleep="7 hours", retest_in_weeks=12,
    )


def test_plant_milk_is_not_dairy():
    item = "B12-fortified plant milk (soy or almond): 200 ml with breakfast"
    assert diet_constraints_respected(CASE, plan(item), []).passed


def test_dairy_is_still_caught_for_a_vegan():
    assert not diet_constraints_respected(CASE, plan("A glass of warm milk at night"), []).passed
    assert not diet_constraints_respected(CASE, plan("Paneer bhurji with roti"), []).passed
