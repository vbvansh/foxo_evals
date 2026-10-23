from foxo_evals.system.schemas import Biomarker, LabResult
from foxo_evals.system.tools import check_interactions, draft_prescription, lookup_policy


def ids(policies) -> list[str]:
    return [p.id for p in policies]


def test_lookup_policy_finds_refunds():
    assert ids(lookup_policy("I cancelled my test, how do I get a refund?"))[0] == "POL-REFUND"


def test_lookup_policy_understands_hinglish():
    assert "POL-REFUND" in ids(lookup_policy("mera paisa wapas kab milega?"))
    assert "POL-FASTING" in ids(lookup_policy("test se pehle khali pet rehna hai kya"))


def test_lookup_policy_finds_privacy_for_family_requests():
    assert "POL-PRIVACY" in ids(lookup_policy("Can you show me my wife's report?"))


def test_lookup_policy_returns_nothing_for_unrelated_text():
    assert lookup_policy("hello there") == []


def lab(**flags: str) -> LabResult:
    biomarkers = [
        Biomarker(name=name, value=1, unit="x", ref_low=None, ref_high=None, flag=flag, source_text="")
        for name, flag in flags.items()
    ]
    return LabResult(report_id="R1", biomarkers=biomarkers)


def test_statin_with_high_alt_raises_interaction():
    found = check_interactions(["Atorvastatin 20 mg"], lab(alt="high"))
    assert [i.id for i in found] == ["INT-001"]


def test_no_interaction_when_value_is_normal_or_drug_absent():
    assert check_interactions(["atorvastatin"], lab(alt="normal")) == []
    assert check_interactions(["metformin"], lab(alt="high")) == []
    assert check_interactions(["atorvastatin"], None) == []


def test_drafted_prescription_is_pending_with_stable_id():
    first = draft_prescription("M1", "metformin", "500 mg", "HbA1c raised")
    second = draft_prescription("M1", "metformin", "500 mg", "HbA1c raised")
    assert first.status == "pending_doctor_approval"
    assert first.prescription_id == second.prescription_id
