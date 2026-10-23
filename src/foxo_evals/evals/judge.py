"""LLM-as-judge, for qualities code can't check (tone, clarity, hidden ingredients).

Each judge prompt lists binary criteria. The judge writes short reasoning first and
then a pass/fail per criterion; the eval passes only if every criterion passes.
Judges run on a different model family from the agents to reduce self-preference.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from foxo_evals.evals.registry import REGISTRY, Severity, Verdict, eval_check
from foxo_evals.llm.client import LLMClient, LLMError
from foxo_evals.prompts import Prompt, load_prompt

JUDGE_PROMPT_VERSION = "v1"


class CriterionVerdict(BaseModel):
    criterion: str
    reasoning: str
    passed: bool


class JudgeOutput(BaseModel):
    criteria: list[CriterionVerdict]


_client: LLMClient | None = None


def set_client(client: LLMClient | None) -> None:
    """The runner sets this once per run, so judge calls share the run's mode and tracer."""
    global _client
    _client = client


def judge_prompt(name: str) -> Prompt:
    return load_prompt(f"judges/{name}", JUDGE_PROMPT_VERSION)


def system_prompt(prompt: Prompt) -> str:
    criteria = "\n".join(f"- {cid}: {text}" for cid, text in prompt.criteria.items())
    return f"{prompt.text}\n\nCriteria (use these ids exactly, one entry each):\n{criteria}"


def ask_judge(name: str, subject: str) -> Verdict:
    if _client is None:
        raise LLMError("judge client is not configured")
    prompt = judge_prompt(name)
    output, _ = _client.complete_json(
        system_prompt(prompt), subject, JudgeOutput, agent=f"judges/{name}", prompt_version=prompt.version
    )
    answered = {c.criterion: c for c in output.criteria}
    missing = [cid for cid in prompt.criteria if cid not in answered]
    if missing:
        return Verdict(False, f"judge skipped criteria {missing}", 0.0)
    failed = [answered[cid] for cid in prompt.criteria if not answered[cid].passed]
    score = round(1 - len(failed) / len(prompt.criteria), 3)
    details = "; ".join(f"{c.criterion}: {c.reasoning}" for c in failed) or "all criteria met"
    return Verdict(not failed, details, score)


def judge_eval(
    *,
    id: str,
    agent: str,
    severity: Severity,
    description: str,
    judge: str,
    applies: Callable[[Any], bool] = lambda case: True,
) -> Callable:
    """Register a judged eval. The decorated function builds the text the judge reads."""

    def register(subject: Callable[[Any, Any], str]) -> Callable:
        def check(case: Any, output: Any, trace: list) -> Verdict:
            return ask_judge(judge, subject(case, output))

        eval_check(id=id, agent=agent, kind="judge", severity=severity, description=description, applies=applies)(check)
        REGISTRY[id].judge = judge
        REGISTRY[id].subject = subject
        return subject

    return register
