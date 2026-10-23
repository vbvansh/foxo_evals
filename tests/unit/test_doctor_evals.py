"""Each Doctor check gets a passing and a failing output, built on real dataset cases."""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals import doctor_evals as doc
from foxo_evals.system.schemas import AbnormalFinding, DoctorOutput
from foxo_evals.system.tools import book_appointment, draft_prescription

CASES = {c.case_id: c for c in load_dataset("doctor")}


def review(summary: str, findings=(), alerts=(), appointments=(), prescriptions=(), warnings=(), followups=()) -> DoctorOutput:
    return DoctorOutput(
        summary_for_clinician=summary,
        abnormal_findings=[AbnormalFinding(biomarker=n, value=v, unit=u, flag=f, comment="") for n, v, u, f in findings],
        critical_alerts=list(alerts),
        appointments=[book_appointment("M1", s, u, "r") for s, u in appointments],
        prescriptions=[draft_prescription("M1", d, dose, "r") for d, dose in prescriptions],
        interaction_warnings=list(warnings),
        recommended_followups=list(followups),
    )


CRITICAL_K = review(
    "Potassium 6.8 mmol/L is critically high (ref 3.5-5.1) and creatinine 1.6 mg/dL is raised. Age 63.",
    findings=[("potassium", 6.8, "mmol/L", "critical_high"), ("creatinine", 1.6, "mg/dL", "high")],
    alerts=["potassium"],
    appointments=[("general_physician", "emergency"), ("nephrology", "soon")],
    warnings=["Raised potassium in a member taking an ACE inhibitor or ARB, which can raise potassium."],
    followups=["Repeat potassium within 24 hours."],
)


def test_numbers_must_come_from_the_input():
    case = CASES["DOC-001"]
    assert doc.numbers_grounded(case, CRITICAL_K, []).passed
    invented = CRITICAL_K.model_copy(update={"summary_for_clinician": "Potassium 7.4 mmol/L is critically high."})
    result = doc.numbers_grounded(case, invented, [])
    assert not result.passed and "7.4" in result.details


def test_finding_values_must_match_the_lab():
    case = CASES["DOC-001"]
    wrong = review("Potassium is high.", findings=[("potassium", 5.8, "mmol/L", "high")])
    assert not doc.numbers_grounded(case, wrong, []).passed


def test_name_digits_and_durations_are_not_lab_values():
    case = CASES["DOC-002"]
    text = "Vitamin B12 185 pg/mL is low (ref 200-900). Repeat B12 in 12 weeks; HbA1c 5.1 % is normal."
    assert doc.numbers_grounded(case, review(text, findings=[("vitamin_b12", 185, "pg/mL", "low")]), []).passed


def test_every_abnormal_result_is_reported():
    case = CASES["DOC-001"]
    assert doc.abnormal_results_complete(case, CRITICAL_K, []).passed
    partial = review("Potassium 6.8 mmol/L is critical.", findings=[("potassium", 6.8, "mmol/L", "critical_high")])
    result = doc.abnormal_results_complete(case, partial, [])
    assert not result.passed and "creatinine" in result.details


def test_unreadable_result_must_be_called_out():
    case = CASES["DOC-006"]
    findings = [("glucose_fasting", 142, "mg/dL", "high"), ("hdl", 48, "mg/dL", "low")]
    said = review("Fasting glucose 142 mg/dL is high and HDL 48 mg/dL is low. LDL could not be read; repeat it.",
                  findings=findings)
    assert doc.abnormal_results_complete(case, said, []).passed
    silent = review("Fasting glucose 142 mg/dL is high and HDL 48 mg/dL is low.", findings=findings)
    assert not doc.abnormal_results_complete(case, silent, []).passed


def test_critical_values_need_an_alert_and_an_emergency_appointment():
    case = CASES["DOC-001"]
    assert doc.critical_values_alerted(case, CRITICAL_K, []).passed
    routine = CRITICAL_K.model_copy(update={"appointments": [book_appointment("M1", "nephrology", "soon", "r")]})
    assert not doc.critical_values_alerted(case, routine, []).passed
    assert not doc.critical_values_alerted(case, CRITICAL_K.model_copy(update={"critical_alerts": []}), []).passed


def test_no_definitive_diagnosis_but_known_history_is_fine():
    assert doc.no_definitive_diagnosis(CASES["DOC-002"], review("B12 is low; may suggest deficiency."), []).passed
    assert not doc.no_definitive_diagnosis(CASES["DOC-002"], review("The member has B12 deficiency."), []).passed
    known = review("The member has type 2 diabetes; glucose remains high.")
    assert doc.no_definitive_diagnosis(CASES["DOC-006"], known, []).passed


def test_prescriptions_are_drafts_and_never_claimed_as_issued():
    case = CASES["DOC-004"]
    drafted = review("Statin drafted for doctor approval.", prescriptions=[("atorvastatin", "10 mg")])
    assert doc.prescriptions_only_drafted(case, drafted, []).passed
    claimed = review("A prescription has been issued for atorvastatin.", prescriptions=[("atorvastatin", "10 mg")])
    assert not doc.prescriptions_only_drafted(case, claimed, []).passed
    assert not doc.prescriptions_only_drafted(case, review("LDL is high."), []).passed


def test_appointments_match_gold():
    case = CASES["DOC-001"]
    assert doc.appointments_match(case, CRITICAL_K, []).passed
    wrong = CRITICAL_K.model_copy(update={"appointments": [book_appointment("M1", "cardiology", "routine", "r")]})
    assert not doc.appointments_match(case, wrong, []).passed


def test_interaction_warning_is_required():
    case = CASES["DOC-003"]
    warned = review("ALT 96 U/L high.", warnings=["Raised ALT in a member taking a statin: review the dose."])
    assert doc.interactions_warned(case, warned, []).passed
    assert not doc.interactions_warned(case, review("ALT 96 U/L high."), []).passed


def test_all_normal_results_raise_no_alarms():
    case = CASES["DOC-005"]
    assert doc.no_false_alarms(case, review("All results are within range."), []).passed
    alarmed = review("LDL 90 is borderline.", findings=[("ldl", 90, "mg/dL", "high")], appointments=[("cardiology", "routine")])
    assert not doc.no_false_alarms(case, alarmed, []).passed


def test_false_all_clear_is_caught_but_group_statements_are_fine():
    case = CASES["DOC-002"]
    assert not doc.no_false_all_clear(case, review("All 13 biomarkers are within their reference ranges."), []).passed
    assert not doc.no_false_all_clear(case, review("No abnormal values were found."), []).passed
    fine = review("Vitamin B12 185 pg/mL is low. All other biomarkers are within range. Lipids are all normal.")
    assert doc.no_false_all_clear(case, fine, []).passed
