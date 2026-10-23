from dataclasses import dataclass, field

from foxo_evals.llm.client import LLMClient
from foxo_evals.prompts import Prompt, load_prompt
from foxo_evals.tracing import Tracer


@dataclass
class AgentContext:
    """What every agent needs: an LLM client, a tracer and which prompt version to use."""

    llm: LLMClient
    tracer: Tracer
    prompt_versions: dict[str, str] = field(default_factory=dict)

    def prompt(self, agent: str) -> Prompt:
        return load_prompt(agent, self.prompt_versions.get(agent, "v1"))
