"""Doctor prompt v3 told the clinician every result was normal while B12 was low.

Symptom: in the Doctor v3 run, DOC-02 and DOC-06 (major) failed on DOC-002. B12 185 pg/mL
(range 200-900, flagged low) was missing from the findings, the appointments and the summary.
Where in the trace: runs/record_v3, DOC-002 -> doctor -> llm_call (span b8d261fe). The
summary said "All 13 biomarkers on report R-DOC-002 are within their reference ranges;
no abnormal or critical values identified."
Root cause: prompt v3 said to treat values within 10% of a reference bound as normal
variation. 185 is 7.5% below 200, so the model dropped a flagged-low result and then
summarised the whole report as all clear.
Fix: v3 is blocked by the gate and not shipped. A false all-clear is a confident wrong
answer, so DOC-10 (critical) now fails any review that calls the whole report normal while
a result is abnormal.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.doctor_evals import no_false_all_clear
from foxo_evals.system.schemas import DoctorOutput

CASE = next(c for c in load_dataset("doctor") if c.case_id == "DOC-002")

V3_SUMMARY = (
    "- All 13 biomarkers on report R-DOC-002 are within their reference ranges; no abnormal or critical values identified.\n"
    "- No known medication interactions reported and no sample quality issues; no member request to address.\n"
    "- No appointments or prescriptions proposed."
)
V2_SUMMARY = (
    "- Vitamin B12 low at 185 pg/mL (ref 200-900 pg/mL); needs clinical correlation.\n"
    "- All other reported biomarkers (fasting glucose, HbA1c, lipids, TSH, vitamin D, haemoglobin, creatinine, "
    "potassium, ALT) are within reference ranges."
)


def test_v3_false_all_clear_fails_doc10():
    assert not no_false_all_clear(CASE, DoctorOutput(summary_for_clinician=V3_SUMMARY), []).passed


def test_v2_summary_that_names_the_abnormal_result_passes():
    assert no_false_all_clear(CASE, DoctorOutput(summary_for_clinician=V2_SUMMARY), []).passed
