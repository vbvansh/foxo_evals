"""Versioned prompts: one markdown file per agent per version, with YAML frontmatter."""

from datetime import date
from functools import lru_cache

import yaml
from pydantic import BaseModel

from foxo_evals.config import PROMPTS_DIR


class Prompt(BaseModel):
    name: str
    version: str
    date: date
    changelog: str
    text: str
    # Judge prompts only: criterion id -> what passing means.
    criteria: dict[str, str] = {}


@lru_cache
def load_prompt(name: str, version: str) -> Prompt:
    """Load prompts/<name>/<version>.md. Name can be nested, e.g. 'judges/no_diagnosis'."""
    path = PROMPTS_DIR / name / f"{version}.md"
    # Normalise line endings: the prompt text is part of the recording key.
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter)
    if meta["version"] != version:
        raise ValueError(f"{path} says version {meta['version']}, expected {version}")
    return Prompt(name=name, text=body.strip(), **meta)
