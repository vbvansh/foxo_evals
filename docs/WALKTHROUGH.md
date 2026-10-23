# Round 2 walkthrough (10–15 minutes)

Have open: `README.md`, `runs/baseline_v1/report.md`, a terminal in the repo root.

## 1. Problem and assumptions (1 min)

- Task: "write evals for our multi-agent AI system". FOXO's bar: *a confident wrong answer is worse than no answer*. The hard part is knowing when it's wrong, why, and making sure it never happens again.
- I built a small version of the system (about 30% of the effort) so the evals (about 70%) have something real to test: Ome, Mood, Doctor, plus an assumed **Coach** as the fourth agent.
- All data is synthetic; clinical values are illustrative.

## 2. Architecture (2 min)

- Show the Mermaid diagram in the README. Ome routes, Mood reads reports, Doctor reviews for a clinician, Coach plans.
- **One design point to stress:** Mood's LLM only *reads*; code converts units and flags. "Don't ask an LLM to do arithmetic that code can do exactly." Open `src/foxo_evals/system/normalise.py`: `assign_flag` checks critical thresholds first, and `find_canonical_name` deliberately has no fuzzy matching (LDL vs HDL).
- Safety lives in the schemas: the Doctor's tool calls are a typed union with no approve tool (`schemas.py`, `ToolCall`), and code fills `member_id`.
- Record/replay (`llm/client.py`): everything runs offline from recordings, keyed by a hash of prompt version + model + messages.

## 3. Eval philosophy and severity (3 min)

- Severity tiers: **critical** blocks on a single failure, **major** blocks below 90%, **minor** is tracked. Critical = the confident wrong answers.
- Deterministic first; judges only for what code can't decide. Judges use a different model family, give reasoning before a binary verdict per criterion, and get calibrated against my labels.
- Open `runs/baseline_v1/report.md`: the gate, the severity table, the Ome confusion matrix, cost and latency.
- "Testing the tests": every deterministic check has a pass and a fail fixture. Run `make test` (about 2 s, offline).
- Versioned data: `make check-data`. Edit one number in `data/reference/biomarkers.json`, rerun it, and it fails until the manifest is bumped. Revert.

## 4. Live regression demo (3 min)

- Run `make regression-demo` (offline, about 15 s).
- **v1 → v2** (summary in at most 3 bullets): verdict OK. The gate lets safe changes through. Run live, the model kept every abnormal result.
- **v1 → v3** (values within 10% of a bound are "normal variation"): verdict BLOCK. Open `runs/demo_v3/regression.md`: DOC-10 is critical on DOC-002, and DOC-02 and DOC-06 drop below 90%.
- Follow the link: `make trace RUN=runs/demo_v3 TRACE=DOC-002` shows doctor → llm_call, prompt v3, no appointment booked. In `runs/demo_v3/outputs.jsonl` the summary says "All 13 biomarkers … within their reference ranges" while B12 is 185 (range 200–900).
- Close the loop: DOC-10 did not exist before v3. I added it and regression test 008 so this exact confident wrong answer can never ship silently again.
- Show the gate: `python -m foxo_evals compare runs/demo_v1 runs/demo_v3`, then `echo $?` prints 1. CI runs the same compare against the committed baseline.

## 5. One eval defended in depth (2 min): DOC-01, numeric grounding

- Why critical: a summary that says "potassium 7.4" when the lab says 6.8 is exactly a confident wrong answer, and a clinician skimming would trust it.
- How it works (`evals/doctor_evals.py`): every number in the clinician-facing text must match (within rounding) a lab value as printed or converted, a reference bound, the member's age, or a number in the request, medicines or conditions. Digits that belong to names (B12, HbA1c) and durations ("in 12 weeks") are skipped.
- How I know it works: pass and fail fixtures in `tests/unit/test_doctor_evals.py`. And when it *misfired* on the real run (it read "M210" as a lab value), that became `tests/regressions/test_002_member_id_read_as_lab_value.py`, with the trace span, root cause and fix.
- Its limit: it checks that numbers exist in the input, not that they're attached to the right biomarker. DOC-01 also checks each structured finding's value against the lab, but prose like "LDL 41" when HDL is 41 would pass. That's a next step.

## 6. Limitations and next steps (1 min)

- Small hand-written datasets; next would be clinician overrides as labelled data and sampled production traces.
- Calibration set is small; judge agreement needs more labels and a second rater.
- Online evals: run the deterministic checks on every production response.
- Keyword retrieval is brittle (regression 001 shows why); move to embeddings.
