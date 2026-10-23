"""Typed inputs and outputs for every agent.

Every handoff between agents is one of these models, so a malformed handoff
fails validation at the boundary instead of confusing the next agent.
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Sex = Literal["male", "female"]
Diet = Literal["vegetarian", "eggetarian", "jain", "vegan", "non_vegetarian"]
Intent = Literal[
    "support", "lab_booking", "report_question", "symptom_concern", "prescription_request", "other"
]
Urgency = Literal["routine", "soon", "emergency"]
Handoff = Literal["none", "mood", "doctor", "coach", "human_clinician"]
Flag = Literal["low", "normal", "high", "critical_low", "critical_high"]
Comparator = Literal["=", "<", ">"]


class MemberProfile(BaseModel):
    member_id: str
    name: str | None = None
    phone: str | None = None
    age: int
    sex: Sex
    conditions: list[str] = []
    medications: list[str] = []
    diet: Diet
    allergies: list[str] = []


# Ome

class MemberMessage(BaseModel):
    member_id: str
    text: str
    context: str | None = None


class OmeResponse(BaseModel):
    intent: Intent
    urgency: Urgency
    reply_text: str
    handoff_to: Handoff
    escalated: bool
    escalation_reason: str | None = None
    cited_policy_ids: list[str] = []


# Mood

class LabReportRaw(BaseModel):
    report_id: str
    member_id: str
    member_sex: Sex
    member_age: int
    collected_at: date
    raw_text: str


class RawBiomarker(BaseModel):
    """What the LLM reads off the report, copied as printed. No conversion, no cleanup."""

    name_as_printed: str
    value_as_printed: str
    unit_as_printed: str


class MoodExtraction(BaseModel):
    biomarkers: list[RawBiomarker]


class Biomarker(BaseModel):
    name: str
    value: float
    comparator: Comparator = "="
    unit: str
    ref_low: float | None
    ref_high: float | None
    flag: Flag
    source_text: str


class LabResult(BaseModel):
    report_id: str
    biomarkers: list[Biomarker]
    quality_issues: list[str] = []
    unparsed_items: list[str] = []


# Doctor

class BookAppointmentCall(BaseModel):
    tool: Literal["book_appointment"]
    specialty: str
    urgency: Urgency
    reason: str


class DraftPrescriptionCall(BaseModel):
    tool: Literal["draft_prescription"]
    drug: str
    dose: str
    reason: str


# There is deliberately no "approve_prescription" tool: approval is human-only.
ToolCall = Annotated[BookAppointmentCall | DraftPrescriptionCall, Field(discriminator="tool")]


class Appointment(BaseModel):
    appointment_id: str
    member_id: str
    specialty: str
    urgency: Urgency
    reason: str
    status: Literal["booked"] = "booked"


class Prescription(BaseModel):
    prescription_id: str
    member_id: str
    drug: str
    dose: str
    reason: str
    status: Literal["pending_doctor_approval"] = "pending_doctor_approval"


class DoctorInput(BaseModel):
    profile: MemberProfile
    lab_result: LabResult | None = None
    member_request: str | None = None


class AbnormalFinding(BaseModel):
    biomarker: str
    value: float
    unit: str
    flag: Flag
    comment: str


class DoctorDraft(BaseModel):
    """What the Doctor LLM returns. Tool calls are requests; the agent code executes them."""

    summary_for_clinician: str
    abnormal_findings: list[AbnormalFinding] = []
    critical_alerts: list[str] = []
    interaction_warnings: list[str] = []
    recommended_followups: list[str] = []
    lifestyle_restrictions: list[str] = []
    tool_calls: list[ToolCall] = []


class DoctorOutput(DoctorDraft):
    appointments: list[Appointment] = []
    prescriptions: list[Prescription] = []


# Coach

class CoachInput(BaseModel):
    profile: MemberProfile
    doctor_output: DoctorOutput


class DietRecommendation(BaseModel):
    item: str
    rationale: str
    linked_biomarker: str


class CoachPlan(BaseModel):
    focus_areas: list[str]
    # Foods to eat and foods to avoid are kept apart so the diet-constraint
    # check can scan recommendations without tripping on "avoid sugary drinks".
    diet_recommendations: list[DietRecommendation]
    foods_to_avoid: list[str] = []
    activity: str
    sleep: str
    retest_in_weeks: int
