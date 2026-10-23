"""Coach: a diet and lifestyle plan linked to the Doctor's findings."""

import json

from foxo_evals.system.context import AgentContext
from foxo_evals.system.schemas import CoachInput, CoachPlan


def build_user_message(coach_input: CoachInput) -> str:
    doctor = coach_input.doctor_output
    # The coach sees only what it needs. No medicines or prescription drafts,
    # so it has nothing to give medication advice about.
    payload = {
        "member_profile": coach_input.profile.model_dump(
            include={"age", "sex", "diet", "allergies", "conditions"}
        ),
        "doctor_review": {
            "abnormal_findings": [f.model_dump(mode="json") for f in doctor.abnormal_findings],
            "lifestyle_restrictions": doctor.lifestyle_restrictions,
            "recommended_followups": doctor.recommended_followups,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_coach(coach_input: CoachInput, ctx: AgentContext) -> CoachPlan:
    prompt = ctx.prompt("coach")
    with ctx.tracer.span("coach", "agent", input=coach_input, prompt_version=prompt.version) as span:
        plan, _ = ctx.llm.complete_json(
            prompt.text,
            build_user_message(coach_input),
            CoachPlan,
            agent="coach",
            prompt_version=prompt.version,
        )
        span.set_output(plan)
    return plan
