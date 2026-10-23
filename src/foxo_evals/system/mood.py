"""Mood: the lab-facing agent. Turns a messy lab report into a structured LabResult.

Two steps on purpose. The LLM only reads (copies names, values and units as
printed). Code converts units, looks up ranges and assigns flags.
"""

from foxo_evals.system.context import AgentContext
from foxo_evals.system.normalise import normalise_report
from foxo_evals.system.schemas import LabReportRaw, LabResult, MoodExtraction


def run_mood(report: LabReportRaw, ctx: AgentContext) -> LabResult:
    prompt = ctx.prompt("mood")
    with ctx.tracer.span("mood", "agent", input=report, prompt_version=prompt.version) as span:
        extraction, _ = ctx.llm.complete_json(
            prompt.text,
            report.raw_text,
            MoodExtraction,
            agent="mood",
            prompt_version=prompt.version,
        )
        with ctx.tracer.span("mood", "normalise_units", input=extraction) as step:
            result = normalise_report(report, extraction.biomarkers)
            step.set_output(result)
        span.set_output(result)
    return result
