"""The digits of a member id were read as an invented lab value.

Symptom: DOC-01 (critical) failed on DOC-010 with "numbers not in the input: [210.0]".
Where in the trace: DOC-010 -> doctor -> llm_call (span b1d3e5a5). The summary began
"Member M210 (45, female)".
Root cause: the number pattern only refused digits that follow another digit or a dot,
so "M210" produced 210. The model did nothing wrong; the eval did.
Fix: digits glued to a letter ("M210", "B12", "HbA1c") are treated as part of a name.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.doctor_evals import numbers_grounded
from foxo_evals.evals.text import numbers_in
from foxo_evals.system.schemas import DoctorOutput

SUMMARY = (
    "Member M210 (45, female) with known hypothyroidism on levothyroxine 50 mcg. TSH is elevated at "
    "7.8 mIU/L (ref 0.4-4.0). Vitamin D (25-OH) is low at 18 ng/mL (ref 30-100). Haemoglobin is "
    "12.4 g/dL (ref 12.0-15.5) and within range."
)


def test_member_id_digits_are_not_numbers():
    assert numbers_in("Member M210") == []
    assert numbers_in("TSH 7.8 and B12 185") == [7.8, 185]


def test_doc01_passes_on_the_real_summary_but_still_catches_invented_values():
    case = next(c for c in load_dataset("doctor") if c.case_id == "DOC-010")
    assert numbers_grounded(case, DoctorOutput(summary_for_clinician=SUMMARY), []).passed
    invented = DoctorOutput(summary_for_clinician=SUMMARY.replace("7.8 mIU/L", "9.6 mIU/L"))
    assert not numbers_grounded(case, invented, []).passed
