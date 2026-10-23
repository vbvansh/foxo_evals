"""Each Mood check gets a passing and a failing output.

Passing outputs come from a perfect extraction run through the real normaliser;
failing ones change one thing, the way a model mistake would.
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals import mood_evals as mood
from foxo_evals.system.normalise import normalise_report
from foxo_evals.system.schemas import LabResult, RawBiomarker

CASES = {c.case_id: c for c in load_dataset("mood")}


def raw(name: str, value: str, unit: str) -> RawBiomarker:
    return RawBiomarker(name_as_printed=name, value_as_printed=value, unit_as_printed=unit)


def perfect(case_id: str) -> LabResult:
    readings = {
        "MOOD-006": [raw("Sodium", "138", "mmol/L"), raw("Potassium", "6.8", "mmol/L"), raw("Creatinine", "2.1", "mg/dL")],
        "MOOD-007": [raw("Total Cholesterol", "214", "mg/dL"), raw("LDL Cholesterol", "##.#", "mg/dL"),
                     raw("HDL Cholesterol", "52", "mg/dL"), raw("Triglycerides", "160", "mg/dL")],
        "MOOD-008": [raw("Potassium", "5.9", "mmol/L"), raw("Creatinine", "1.0", "mg/dL")],
    }[case_id]
    return normalise_report(CASES[case_id].report, readings)


def with_biomarker(result: LabResult, name: str, **changes) -> LabResult:
    biomarkers = [b.model_copy(update=changes) if b.name == name else b for b in result.biomarkers]
    return result.model_copy(update={"biomarkers": biomarkers})


def test_perfect_extraction_passes_every_check():
    for case_id in ("MOOD-006", "MOOD-007", "MOOD-008"):
        case, output = CASES[case_id], perfect(case_id)
        for check in (mood.extraction_precision_recall, mood.values_grounded, mood.converted_values_correct,
                      mood.flags_correct, mood.quality_issues_detected, mood.output_schema_valid):
            assert check(case, output, []).passed, (case_id, check.__name__)


def test_precision_and_recall():
    case, output = CASES["MOOD-006"], perfect("MOOD-006")
    assert mood.extraction_precision_recall(case, output, []).score == 1.0
    missing = output.model_copy(update={"biomarkers": output.biomarkers[:1]})
    result = mood.extraction_precision_recall(case, missing, [])
    assert not result.passed and "recall 0.50" in result.details and result.score < 1


def test_hallucinated_value_is_not_grounded():
    case, output = CASES["MOOD-007"], perfect("MOOD-007")
    invented = with_biomarker(output, "hdl", source=raw("HDL Cholesterol", "48", "mg/dL"), value=48)
    result = mood.values_grounded(case, invented, [])
    assert not result.passed and "48" in result.details


def test_wrong_conversion_is_caught():
    case, output = CASES["MOOD-006"], perfect("MOOD-006")
    assert not mood.converted_values_correct(case, with_biomarker(output, "creatinine", value=21.0), []).passed


def test_critical_flag_must_not_be_downgraded():
    case, output = CASES["MOOD-006"], perfect("MOOD-006")
    assert mood.critical_flags_recalled(case, output, []).passed
    downgraded = with_biomarker(output, "potassium", flag="high")
    assert not mood.critical_flags_recalled(case, downgraded, []).passed
    assert not mood.flags_correct(case, downgraded, []).passed


def test_unreadable_value_must_not_be_invented():
    case, output = CASES["MOOD-007"], perfect("MOOD-007")
    assert mood.unreadable_not_guessed(case, output, []).passed
    guessed = normalise_report(case.report, [raw("LDL Cholesterol", "120", "mg/dL")])
    result = mood.unreadable_not_guessed(case, guessed, [])
    assert not result.passed and "invented a value for ldl" in result.details


def test_quality_issue_must_be_detected():
    case, output = CASES["MOOD-008"], perfect("MOOD-008")
    assert mood.quality_issues_detected(case, output, []).passed
    assert not mood.quality_issues_detected(case, output.model_copy(update={"quality_issues": []}), []).passed


def test_output_must_match_reference_units_and_ranges():
    case, output = CASES["MOOD-008"], perfect("MOOD-008")
    assert not mood.output_schema_valid(case, with_biomarker(output, "potassium", unit="mg/dL"), []).passed
    assert not mood.output_schema_valid(case, with_biomarker(output, "potassium", ref_high=6.0), []).passed
