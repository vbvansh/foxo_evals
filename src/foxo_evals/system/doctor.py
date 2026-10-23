"""Doctor: reviews lab results for a clinician, books appointments, drafts prescriptions."""

import json

from foxo_evals.system.context import AgentContext
from foxo_evals.system.schemas import BookAppointmentCall, DoctorDraft, DoctorInput, DoctorOutput
from foxo_evals.system.tools import Interaction, book_appointment, check_interactions, draft_prescription


def build_user_message(doctor_input: DoctorInput, interactions: list[Interaction]) -> str:
    lab = doctor_input.lab_result
    payload = {
        # The model doesn't need the member's name or phone number to review labs.
        "member_profile": doctor_input.profile.model_dump(exclude={"full_name", "phone"}),
        "lab_result": lab.model_dump(mode="json") if lab else None,
        "known_interactions": [i.warning for i in interactions],
        "member_request": doctor_input.member_request,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_doctor(doctor_input: DoctorInput, ctx: AgentContext) -> DoctorOutput:
    prompt = ctx.prompt("doctor")
    member_id = doctor_input.profile.member_id
    with ctx.tracer.span("doctor", "agent", input=doctor_input, prompt_version=prompt.version) as span:
        with ctx.tracer.span("doctor", "tool:check_interactions", input=doctor_input.profile.medications) as tool:
            interactions = check_interactions(doctor_input.profile.medications, doctor_input.lab_result)
            tool.set_output([i.id for i in interactions])

        draft, _ = ctx.llm.complete_json(
            prompt.text,
            build_user_message(doctor_input, interactions),
            DoctorDraft,
            agent="doctor",
            prompt_version=prompt.version,
        )

        # The model only requests tool calls. Code executes them, and fills in
        # member_id itself so the model can never act on another member's record.
        appointments, prescriptions = [], []
        for call in draft.tool_calls:
            with ctx.tracer.span("doctor", f"tool:{call.tool}", input=call) as tool:
                if isinstance(call, BookAppointmentCall):
                    result = book_appointment(member_id, call.specialty, call.urgency, call.reason)
                    appointments.append(result)
                else:
                    result = draft_prescription(member_id, call.drug, call.dose, call.reason)
                    prescriptions.append(result)
                tool.set_output(result)

        output = DoctorOutput(**draft.model_dump(), appointments=appointments, prescriptions=prescriptions)
        span.set_output(output)
    return output
