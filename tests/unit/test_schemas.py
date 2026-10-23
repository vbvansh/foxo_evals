import pytest
from pydantic import ValidationError

from foxo_evals.system.schemas import (
    BookAppointmentCall,
    DoctorDraft,
    DraftPrescriptionCall,
    OmeResponse,
    Prescription,
)


def test_tool_calls_are_parsed_by_tool_name():
    draft = DoctorDraft.model_validate(
        {
            "summary_for_clinician": "LDL raised.",
            "tool_calls": [
                {"tool": "book_appointment", "specialty": "cardiology", "urgency": "soon", "reason": "LDL"},
                {"tool": "draft_prescription", "drug": "atorvastatin", "dose": "10 mg", "reason": "LDL"},
            ],
        }
    )
    assert isinstance(draft.tool_calls[0], BookAppointmentCall)
    assert isinstance(draft.tool_calls[1], DraftPrescriptionCall)


def test_there_is_no_approve_tool():
    with pytest.raises(ValidationError):
        DoctorDraft.model_validate(
            {"summary_for_clinician": "x", "tool_calls": [{"tool": "approve_prescription", "id": "P1"}]}
        )


def test_prescriptions_can_only_be_pending():
    rx = Prescription(prescription_id="P1", member_id="M1", drug="metformin", dose="500 mg", reason="HbA1c")
    assert rx.status == "pending_doctor_approval"
    with pytest.raises(ValidationError):
        Prescription(
            prescription_id="P1", member_id="M1", drug="metformin", dose="500 mg",
            reason="HbA1c", status="approved",
        )


def test_unknown_handoff_target_is_rejected():
    with pytest.raises(ValidationError):
        OmeResponse(
            intent="support", urgency="routine", reply_text="Hi", handoff_to="pharmacist", escalated=False
        )
