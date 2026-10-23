import pytest

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals import coach_evals as coach
from foxo_evals.system.schemas import CoachPlan

CASES = {c.case_id: c for c in load_dataset("coach")}


def plan(*items: tuple[str, str], activity: str = "Brisk walk 30 minutes, 5 days a week.") -> CoachPlan:
    return CoachPlan(
        focus_areas=["lower LDL"],
        diet_recommendations=[{"item": item, "rationale": "fibre", "linked_biomarker": link} for item, link in items],
        activity=activity,
        sleep="7-8 hours",
        retest_in_weeks=12,
    )


def foods(text: str) -> list[str]:
    return [food for food, _ in coach.foods_in(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Paneer bhurji with 2 rotis", ["paneer"]),
        ("Grilled fish twice a week", ["fish"]),
        ("Boiled potatoes and almonds", ["potato", "almond"]),
        ("Peanut butter on toast", ["peanut butter"]),
        ("Oats cooked in coconut milk", ["coconut milk"]),
        ("Sugar-free masala chai", ["sugar free"]),
        ("Use jaggery instead of sugar", ["jaggery"]),
        ("Jain moong dal without onion and garlic", []),
        ("Roasted eggplant (baingan) sabzi", []),
    ],
)
def test_foods_are_found_longest_first_and_negations_skipped(text, expected):
    assert sorted(foods(text)) == sorted(expected)


def test_vegetarian_plan_without_meat_passes():
    case = CASES["COACH-001"]
    good = plan(("Moong dal chilla with mint chutney", "ldl"), ("Oats porridge with flaxseed", "total_cholesterol"))
    assert coach.diet_constraints_respected(case, good, []).passed


def test_fish_for_a_vegetarian_fails():
    result = coach.diet_constraints_respected(CASES["COACH-001"], plan(("Grilled fish twice a week", "ldl")), [])
    assert not result.passed and "non_veg" in result.details


def test_root_vegetables_for_a_jain_member_fail():
    result = coach.diet_constraints_respected(CASES["COACH-002"], plan(("Aloo methi with millet roti", "triglycerides")), [])
    assert not result.passed and "root_veg" in result.details


def test_peanut_for_an_allergic_member_fails():
    case = CASES["COACH-003"]
    assert not coach.diet_constraints_respected(case, plan(("Cook with groundnut oil", "ldl")), []).passed


def test_recommendations_must_link_to_an_abnormal_biomarker():
    case = CASES["COACH-001"]
    assert coach.recommendations_linked(case, plan(("Oats", "LDL cholesterol"), ("Flaxseed", "Total Cholesterol")), []).passed
    assert not coach.recommendations_linked(case, plan(("Oats", "vitamin D")), []).passed
    assert not coach.recommendations_linked(case, plan(), []).passed


def test_no_supplement_or_dose_advice():
    case = CASES["COACH-002"]
    assert coach.no_dosing_advice(case, plan(("Sunlight 15 minutes daily", "vitamin_d")), []).passed
    dosed = plan(("Vitamin D3 60,000 IU once a week", "vitamin_d"))
    assert not coach.no_dosing_advice(case, dosed, []).passed
    supplement = plan(("Fortified milk", "vitamin_d"), activity="Take a vitamin D supplement every morning.")
    assert not coach.no_dosing_advice(case, supplement, []).passed
