"""Ome: the member-facing agent. Classifies the message, routes it, and answers support questions."""

from foxo_evals.system.context import AgentContext
from foxo_evals.system.schemas import MemberMessage, OmeResponse
from foxo_evals.system.tools import Policy, escalate_to_human, lookup_policy


def build_user_message(message: MemberMessage, policies: list[Policy]) -> str:
    lines = [f"Member ID: {message.member_id}", f"Message: {message.text}"]
    if message.context:
        lines.append(f"Context: {message.context}")
    lines.append("Policies:")
    lines += [f"[{p.id}] {p.title}: {p.text}" for p in policies] or ["(no matching policy)"]
    return "\n".join(lines)


def run_ome(message: MemberMessage, ctx: AgentContext) -> OmeResponse:
    prompt = ctx.prompt("ome")
    with ctx.tracer.span("ome", "agent", input=message, prompt_version=prompt.version) as span:
        # Policies are looked up before the model answers, so support answers
        # come from the knowledge base rather than the model's memory.
        with ctx.tracer.span("ome", "tool:lookup_policy", input=message.text) as tool:
            policies = lookup_policy(message.text)
            tool.set_output([p.id for p in policies])

        response, _ = ctx.llm.complete_json(
            prompt.text,
            build_user_message(message, policies),
            OmeResponse,
            agent="ome",
            prompt_version=prompt.version,
        )

        if response.escalated:
            with ctx.tracer.span("ome", "tool:escalate_to_human", input=response.escalation_reason) as tool:
                tool.set_output(escalate_to_human(message.member_id, response.escalation_reason or ""))
        span.set_output(response)
    return response
