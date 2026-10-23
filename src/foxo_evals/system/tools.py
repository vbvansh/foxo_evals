"""Tools the agents can use. All simulated in memory: no real bookings or prescriptions.

Note what is missing: there is no tool to approve a prescription. Approval is a
human action outside the system, so no prompt or model change can ever approve one.
"""

import hashlib
import json
import re
from functools import lru_cache

from pydantic import BaseModel

from foxo_evals.config import KB_DIR, REFERENCE_DIR
from foxo_evals.system.schemas import Appointment, Flag, LabResult, Prescription, Specialty, Urgency


class Policy(BaseModel):
    id: str
    title: str
    text: str
    keywords: list[str]


class Interaction(BaseModel):
    id: str
    drug_class: str
    drugs: list[str]
    biomarker: str
    flags: list[Flag]
    warning: str


class Escalation(BaseModel):
    ticket_id: str
    member_id: str
    reason: str
    status: str = "open"


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache
def load_policies() -> list[Policy]:
    return [Policy.model_validate(row) for row in read_jsonl(KB_DIR / "policies.jsonl")]


@lru_cache
def load_interactions() -> list[Interaction]:
    return [Interaction.model_validate(row) for row in read_jsonl(REFERENCE_DIR / "interactions.jsonl")]


def short_id(prefix: str, *parts: str) -> str:
    """Deterministic ids, so replayed runs produce identical outputs."""
    return f"{prefix}-{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:6].upper()}"


def lookup_policy(query: str, top_k: int = 2) -> list[Policy]:
    """Rank policies by how many of their keywords appear in the query.

    The knowledge base is six entries, so keyword overlap is enough and is easy
    to debug. A real KB would need embeddings.
    """
    words = set(re.findall(r"[a-z]+", query.lower()))
    scored = []
    for policy in load_policies():
        score = sum(1 for keyword in policy.keywords if set(keyword.split()) <= words)
        if score:
            scored.append((score, policy))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [policy for _, policy in scored[:top_k]]


def escalate_to_human(member_id: str, reason: str) -> Escalation:
    return Escalation(ticket_id=short_id("ESC", member_id, reason), member_id=member_id, reason=reason)


def book_appointment(member_id: str, specialty: Specialty, urgency: Urgency, reason: str) -> Appointment:
    return Appointment(
        appointment_id=short_id("APT", member_id, specialty, reason),
        member_id=member_id,
        specialty=specialty,
        urgency=urgency,
        reason=reason,
    )


def draft_prescription(member_id: str, drug: str, dose: str, reason: str) -> Prescription:
    """Creates a draft only. Status is always pending_doctor_approval."""
    return Prescription(
        prescription_id=short_id("RX", member_id, drug, dose),
        member_id=member_id,
        drug=drug,
        dose=dose,
        reason=reason,
    )


def check_interactions(medications: list[str], lab_result: LabResult | None) -> list[Interaction]:
    """Match the member's medicines against abnormal results. A table lookup, so code does it."""
    if lab_result is None:
        return []
    flags = {b.name: b.flag for b in lab_result.biomarkers}
    meds = " ".join(medications).lower()
    return [
        row
        for row in load_interactions()
        if flags.get(row.biomarker) in row.flags and any(drug in meds for drug in row.drugs)
    ]
