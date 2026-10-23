"""A report id was read as an invented lab value.

Symptom: DOC-01 (critical) failed on DOC-002 and DOC-005 in the Doctor v3 run with
"numbers not in the input: [2.0]" and "[5.0]".
Where in the trace: runs/record_v3, DOC-002 -> doctor -> llm_call (span b8d261fe). The
summary said "All 13 biomarkers on report R-DOC-002 are within their reference ranges".
Root cause: regression 002 taught DOC-01 to skip digits glued to a letter ("M210"), but in
"R-DOC-002" the digits follow a hyphen, so "002" was read as the number 2.
Fix: ids made of letters, hyphens and digits ("R-DOC-002", "INT-001") count as names.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.doctor_evals import numbers_grounded
from foxo_evals.system.schemas import DoctorOutput

CASES = {c.case_id: c for c in load_dataset("doctor")}


def test_report_and_interaction_ids_are_not_numbers():
    summary = "All 13 biomarkers on report R-DOC-005 are within their reference ranges (see INT-001)."
    assert numbers_grounded(CASES["DOC-005"], DoctorOutput(summary_for_clinician=summary), []).passed


def test_a_real_invented_value_is_still_caught():
    summary = "Report R-DOC-005: potassium 6.2 mmol/L."
    assert not numbers_grounded(CASES["DOC-005"], DoctorOutput(summary_for_clinician=summary), []).passed
