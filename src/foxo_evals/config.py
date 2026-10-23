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
CALIBRATION_DIR = DATA_DIR / "calibration"
REPORTS_DIR = ROOT / "reports"

load_dotenv(ROOT / ".env")

# live: call the API. record: replay what exists, call the API for anything
# missing and save it. replay: recordings only, no network, no key needed.
MODE = os.getenv("FOXO_MODE", "replay")

# OpenCode Go subscription endpoint (OpenAI-compatible chat completions).
OPENCODE_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
AGENT_MODEL = "deepseek-v4.1-flash"
# A different model family judges the agents, to reduce self-preference bias.
JUDGE_MODEL = "glm-5.3"

# USD per 1M tokens (input, output), from the OpenCode Go price table on
# 2026-10-23 (DeepSeek at its peak-hour rate). Update if outdated.
PRICES = {
    "deepseek-v4.1-flash": (0.30, 1.20),
    "glm-5.3": (1.40, 4.40),
}

# Per-journey budgets for SYS-04 and SYS-05. The agents use a reasoning model and
# a clinician reviews the output before the member sees it, so this is an async
# budget, not chat latency.
JOURNEY_LATENCY_BUDGET_MS = 180_000
JOURNEY_COST_BUDGET_USD = 0.01
