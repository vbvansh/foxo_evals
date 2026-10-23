"""Routes one member message through the agents.

Plain Python on purpose: the routing is a handful of if statements, so it is easy
to read, test and trace. Every object handed to the next agent is re-validated.
"""

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from foxo_evals.llm.client import LLMError
from foxo_evals.system.coach import run_coach
from foxo_evals.system.context import AgentContext
from foxo_evals.system.doctor import run_doctor
from foxo_evals.system.mood import run_mood
from foxo_evals.system.ome import run_ome
from foxo_evals.system.schemas import CoachInput, DoctorInput, JourneyInput, JourneyResult

T = TypeVar("T", bound=BaseModel)


def handoff(ctx: AgentContext, to_agent: str, obj: T) -> T:
    """Round-trip the object through JSON, as if it crossed a service boundary.

    Catches anything that was changed after it was built and no longer fits its schema.
    """
    with ctx.tracer.span("orchestrator", f"handoff:{to_agent}", input=type(obj).__name__):
        return type(obj).model_validate_json(obj.model_dump_json())


def run_member_journey(journey: JourneyInput, ctx: AgentContext) -> JourneyResult:
    """Ome first, then Mood, Doctor and Coach as needed. A failing agent stops the journey
    and is recorded in result.error instead of crashing the eval run."""
    result = JourneyResult()
    with ctx.tracer.span("orchestrator", "journey", input=journey.message) as span:
        try:
            _route(journey, ctx, result)
        except (LLMError, ValidationError) as error:
            result.error = f"{type(error).__name__}: {error}"
        span.set_output({"trajectory": result.trajectory, "error": result.error})
    return result


def _route(journey: JourneyInput, ctx: AgentContext, result: JourneyResult) -> None:
    result.ome = run_ome(journey.message, ctx)
    result.trajectory.append("ome")
    if result.ome.handoff_to in ("none", "human_clinician"):
        return

    if journey.lab_report:
        result.lab_result = run_mood(handoff(ctx, "mood", journey.lab_report), ctx)
        result.trajectory.append("mood")
    if result.lab_result is None and result.ome.handoff_to != "doctor":
        return  # e.g. a lab booking with no report yet: nothing for the Doctor to review

    doctor_input = DoctorInput(
        profile=journey.profile, lab_result=result.lab_result, member_request=journey.message.text
    )
    result.doctor = run_doctor(handoff(ctx, "doctor", doctor_input), ctx)
    result.trajectory.append("doctor")

    # No lifestyle plan without lab results, or while a critical value waits for a doctor.
    if result.lab_result is None or result.doctor.critical_alerts:
        return
    coach_input = CoachInput(profile=journey.profile, doctor_output=result.doctor)
    result.coach = run_coach(handoff(ctx, "coach", coach_input), ctx)
    result.trajectory.append("coach")
