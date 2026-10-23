"""Refund question went unanswered because the policy lookup missed "cancelled".

Symptom: OME-05 failed on OME-005. Ome replied "I'll check with the team" instead of
quoting the refund policy.
Where in the trace: OME-005 -> ome -> tool:lookup_policy (span 3195129e) returned
["POL-COLLECTION", "POL-TURNAROUND"]. The eval first pointed at ome -> llm_call
(span 8a2aaf15), where the symptom showed, not where it broke.
Root cause: lookup_policy matched whole words, so the keyword "cancel" never matched
"cancelled". The model then correctly followed its rule for questions no policy covers.
Fix: keyword words now match the start of a query word. OME-05 also checks what the
lookup returned and points at the tool span when the policy was never retrieved.
Date: 2026-10-23
"""

from foxo_evals.datasets.loader import load_dataset
from foxo_evals.evals.ome_evals import policy_answers_grounded
from foxo_evals.system.schemas import OmeResponse
from foxo_evals.system.tools import lookup_policy
from foxo_evals.tracing import Span

MESSAGE = "I cancelled my home collection 6 hours before the slot. How much will I get back and when?"
REPLY = OmeResponse(
    intent="support", urgency="routine", handoff_to="none", escalated=False, cited_policy_ids=[],
    reply_text="I don't have the refund details in front of me, so I'll check with the team.",
)


def span(step: str, output: str = "") -> Span:
    return Span(trace_id="OME-005", span_id=step, parent_span_id=None, agent="ome", step=step, output_summary=output)


def test_refund_policy_is_found_for_cancelled():
    assert "POL-REFUND" in [p.id for p in lookup_policy(MESSAGE)]


def test_ome05_points_at_the_lookup_when_the_policy_was_not_retrieved():
    case = next(c for c in load_dataset("ome") if c.case_id == "OME-005")
    trace = [span("tool:lookup_policy", '["POL-COLLECTION", "POL-TURNAROUND"]'), span("llm_call")]
    verdict = policy_answers_grounded(case, REPLY, trace)
    assert not verdict.passed
    assert verdict.span.step == "tool:lookup_policy"
