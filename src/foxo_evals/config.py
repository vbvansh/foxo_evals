"""Paths and settings shared across the project."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
KB_DIR = DATA_DIR / "kb"
PROMPTS_DIR = ROOT / "prompts"
RECORDINGS_DIR = ROOT / "recordings"
RUNS_DIR = ROOT / "runs"

load_dotenv(ROOT / ".env")

# live: call the API. record: replay what exists, call the API for anything
# missing and save it. replay: recordings only, no network, no key needed.
MODE = os.getenv("FOXO_MODE", "replay")

OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1"
AGENT_MODEL = "deepseek-v4.1-flash"
# A different model family judges the agents, to reduce self-preference bias.
JUDGE_MODEL = "mistral-large-4"

# USD per 1M tokens (input, output), from opencode.ai pricing on 2026-10-23.
# Update if outdated.
PRICES = {
    "deepseek-v4.1-flash": (0.30, 1.20),
    "mistral-large-4": (0.68, 2.09),
}
