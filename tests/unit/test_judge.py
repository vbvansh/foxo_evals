import pytest

from foxo_evals.evals import judge
from foxo_evals.llm.client import LLMError, Usage


class FakeJudgeLLM:
    """Answers every criterion with the given verdicts and remembers the prompt it got."""

    def __init__(self, verdicts: dict[str, bool]):
        self.verdicts = verdicts
        self.system = ""

    def complete_json(self, system, user, schema, *, agent, prompt_version):
        self.system = system
        criteria = [{"criterion": c, "reasoning": f"{c} reasoning", "passed": p} for c, p in self.verdicts.items()]
        return schema.model_validate({"criteria": criteria}), Usage()


@pytest.fixture(autouse=True)
def reset_client():
    yield
    judge.set_client(None)


def test_all_criteria_must_pass():
    criteria = judge.judge_prompt("ome_tone").criteria
    judge.set_client(FakeJudgeLLM({c: True for c in criteria}))
    verdict = judge.ask_judge("ome_tone", "Member's message: hi")
    assert verdict.passed and verdict.score == 1.0


def test_one_failed_criterion_fails_the_eval_with_its_reasoning():
    criteria = list(judge.judge_prompt("ome_tone").criteria)
    judge.set_client(FakeJudgeLLM({c: c != "clear" for c in criteria}))
    verdict = judge.ask_judge("ome_tone", "Member's message: hi")
    assert not verdict.passed
    assert "clear: clear reasoning" in verdict.details
    assert verdict.score == pytest.approx(0.75)


def test_skipped_criteria_fail():
    judge.set_client(FakeJudgeLLM({"empathetic": True}))
    verdict = judge.ask_judge("ome_tone", "x")
    assert not verdict.passed and "skipped" in verdict.details


def test_criteria_from_frontmatter_are_sent_to_the_judge():
    fake = FakeJudgeLLM({c: True for c in judge.judge_prompt("coach_restrictions").criteria})
    judge.set_client(fake)
    judge.ask_judge("coach_restrictions", "x")
    assert "follows_restrictions:" in fake.system


def test_every_judge_prompt_has_criteria():
    from foxo_evals.runner import REGISTRY

    for spec in REGISTRY.values():
        if spec.kind == "judge":
            assert judge.judge_prompt(spec.judge).criteria, spec.id


def test_judging_without_a_client_is_an_llm_error():
    judge.set_client(None)
    with pytest.raises(LLMError):
        judge.ask_judge("ome_tone", "x")
