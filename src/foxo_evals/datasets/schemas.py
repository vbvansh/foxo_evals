"""Schemas for the eval datasets: what each case gives the agent, and its gold labels.

Rows are validated strictly (unknown fields and unknown tags are errors), because a
typo in a dataset should fail loudly, not quietly drop a case out of an eval.
"""

from pydantic import BaseModel, ConfigDict, field_validator

from foxo_evals.system.schemas import (
    CoachInput,
    Comparator,
    DoctorInput,
    Flag,
    Handoff,
    Intent,
    JourneyInput,
    LabReportRaw,
    MemberMessage,
    Specialty,
    Urgency,
)

KNOWN_TAGS = {
    # safety
    "safety:emergency", "safety:self_harm", "safety:injection", "safety:privacy", "safety:diagnosis",
    # Ome routing and language
    "route:support", "route:report", "route:booking", "route:prescription", "route:symptom",
    "route:coach", "route:ambiguous", "kb:grounded", "lang:hinglish",
    # Mood edge cases
    "edge:clean", "edge:unit_mmol", "edge:unit_nmol", "edge:unit_si", "edge:sex_specific",
    "edge:critical", "edge:unreadable", "edge:haemolysed", "edge:duplicate", "edge:less_than",
    "edge:ocr_noise", "edge:unknown_marker",
    # Doctor
    "clinical:critical", "clinical:subtle_abnormal", "clinical:interaction",
    "clinical:prescription_request", "clinical:all_normal", "clinical:missing_values",
    "clinical:haemolysed",
    # Coach
    "diet:vegetarian", "diet:eggetarian", "diet:vegan", "diet:jain", "allergy:peanut",
    "allergy:tree_nut", "condition:diabetes", "restriction:exercise",
    # End to end
    "journey:routine_report", "journey:critical_report", "journey:support", "journey:emergency",
    "journey:prescription", "journey:diet_plan",
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(Strict):
    case_id: str
    description: str
    tags: list[str]
    added_in_version: str

    @field_validator("tags")
    @classmethod
    def tags_are_known(cls, tags: list[str]) -> list[str]:
        unknown = set(tags) - KNOWN_TAGS
        if unknown:
            raise ValueError(f"unknown tags {sorted(unknown)}")
        return tags


class OmeGold(Strict):
    intent: Intent
    handoff_to: Handoff
    escalate: bool
    policy_ids: list[str] = []
    # Data that must never appear in the reply, even inside a refusal
    # (another member's results). Matched anywhere.
    forbidden_data: list[str] = []
    # Claims the reply must not make ("you are healthy"). Ignored when negated
    # in the same clause, so "I can't say you are healthy" is fine.
    forbidden_claims: list[str] = []


class OmeCase(Case):
    message: MemberMessage
    gold: OmeGold


class GoldBiomarker(Strict):
    """Values in canonical units, worked out by hand from the printed report."""

    name: str
    value: float
    unit: str
    flag: Flag
    comparator: Comparator = "="


class MoodGold(Strict):
    biomarkers: list[GoldBiomarker]
    # Names that must land in unparsed_items and must not appear as biomarkers.
    unparsed: list[str] = []
    quality_issues: list[str] = []


class MoodCase(Case):
    report: LabReportRaw
    gold: MoodGold


class ExpectedAppointment(Strict):
    specialty: Specialty
    urgency: Urgency


class DoctorGold(Strict):
    abnormal: list[str]
    critical: list[str] = []
    appointments: list[ExpectedAppointment] = []
    interactions: list[str] = []
    expects_prescription_draft: bool = False
    # Results that are missing or unreadable: the summary must say so, not guess.
    unavailable: list[str] = []


class DoctorCase(Case):
    input: DoctorInput
    gold: DoctorGold


class CoachGold(Strict):
    forbidden_tags: list[str]


class CoachCase(Case):
    input: CoachInput
    gold: CoachGold


class E2EGold(Strict):
    trajectory: list[str]
    critical: list[str] = []
    escalate: bool = False


class E2ECase(Case):
    journey: JourneyInput
    gold: E2EGold


CASE_TYPES: dict[str, type[Case]] = {
    "ome": OmeCase,
    "mood": MoodCase,
    "doctor": DoctorCase,
    "coach": CoachCase,
    "e2e": E2ECase,
}
