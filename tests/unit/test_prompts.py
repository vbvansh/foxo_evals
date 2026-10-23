import pytest

from foxo_evals.config import PROMPTS_DIR
from foxo_evals.prompts import load_prompt

PROMPT_FILES = sorted(PROMPTS_DIR.rglob("v*.md"))


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: str(p.relative_to(PROMPTS_DIR)))
def test_every_prompt_has_valid_frontmatter(path):
    name = path.parent.relative_to(PROMPTS_DIR).as_posix()
    prompt = load_prompt(name, path.stem)
    assert prompt.version == path.stem
    assert prompt.changelog
    assert prompt.text and "---" not in prompt.text.splitlines()[0]


def test_all_agents_have_a_v1_prompt():
    for agent in ("ome", "mood", "doctor", "coach"):
        assert load_prompt(agent, "v1").text
