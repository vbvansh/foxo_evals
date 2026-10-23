# Design decisions

Each entry: the decision, why, and what it costs.

## System under test

- **Plain Python agents, no agent framework.** Each agent is one function with a typed input and output, and the orchestrator is a few `if` statements.
  *Why:* every step is easy to read, test and trace. *Trade-off:* no built-in memory, streaming or parallelism.
- **Mood: the LLM only reads; code converts and flags.** The model copies name, value and unit as printed. `normalise.py` maps names, converts units, looks up sex-specific ranges and assigns flags.
  *Why:* never ask an LLM to do arithmetic or table lookups that code can do exactly. *Trade-off:* the alias list needs maintaining, and an unknown name goes to `unparsed_items`.
- **No fuzzy name matching.** Names must match an alias exactly; the only fallback is common OCR swaps (`0`→o, `1`→l).
  *Why:* "LDL cholesterol" and "HDL cholesterol" are one letter apart. *Trade-off:* a misspelt test name needs a human, which is safer than a wrong guess.
- **Never guess a value.** Unreadable values, missing or unknown units, and duplicates that disagree all go to `unparsed_items`. Duplicates within 2% (the same reading printed in two units) are merged.
- **Typed tool calls, and no approve tool.** The Doctor can request only `book_appointment` or `draft_prescription`. Code executes them and fills in `member_id` itself, and drafts are always `pending_doctor_approval`.
  *Why:* the approval guarantee lives in the schema, not in a prompt.
- **Ome always looks up policies before answering.** It uses keyword retrieval over a six-entry knowledge base.
  *Why:* support answers come from the knowledge base, not model memory. *Trade-off:* keyword matching is brittle. The first real run proved it (regression test 001), and a real KB would need embeddings.
- **The Coach sees only what it needs** (diet, allergies, conditions, findings and restrictions), never medicines or drafts. It does not run while a critical alert waits for a doctor.
- **The Doctor prompt contains an appointment routing table** (lipids go to cardiology, routine, and so on).
  *Why:* without it, "the right specialty" is ambiguous and DOC-06 would be measuring noise.

## LLM client and tracing

- **Record and replay are keyed on a sha256** of agent, prompt version, model and the full messages, including the JSON schema.
  *Why:* tests and evals run offline, for free and identically every time, and a prompt, schema or model change can never reuse an old answer. *Trade-off:* replay can't see provider drift, so live runs still need scheduling.
- **Record mode calls the API only for missing responses.**
  *Why:* re-recording would silently change results that were already reviewed. `prune-recordings` removes files no run uses.
- **Replay reports the latency, tokens and cost measured at record time**, so the latency and cost budgets (SYS-04 and SYS-05) still mean something offline.
- **Trace id = case id, and span ids are hashes of position.** Replays produce identical ids, and a report line like "doctor → llm_call (span 3f9a1c02)" can be looked up with `make trace`.
- **OpenCode Go endpoint.** One key reaches both model families: `deepseek-v4.1-flash` for the agents and `glm-5.3` for the judge.

## Evals

- **Severity tiers come from FOXO's own rule: "a confident wrong answer is worse than no answer."**
  - Critical: any single failure blocks the release. These are wrong numbers, missed critical values, diagnoses, leaked data, unsafe diets and unescalated emergencies.
  - Major: the pass rate must stay at or above 90%. Minor: tracked only.
- **Deterministic checks first; a judge only where code can't decide.** Judges cover tone and clarity, and act as a backup for diagnosis wording and hidden ingredients. They run on a different model family, give reasoning before a binary verdict per criterion, and their prompts are versioned.
- **COACH-02 (doctor's restrictions) is critical but judge-only.**
  *Trade-off:* the release gate depends on the judge being right, which is why calibration against human labels exists.
- **Leaks and claims are matched differently.**
  - `forbidden_data` (another member's numbers) matches anywhere, even inside a refusal.
  - `forbidden_claims` ("you are healthy") is ignored when its own clause is negated.

  *Trade-off:* the negation check is a heuristic. It works by clause, and for food lists a comma doesn't end the negation (regression test 005).
- **An eval points at the span it judged**, and at the tool span when the cause is upstream (OME-05 now blames `tool:lookup_policy` when retrieval missed).
- **Evals have their own tests.** Every deterministic check has a passing and a failing fixture. Every bug in an eval found on real output has a regression test.

## Datasets

- **The manifest hashes datasets, reference data and the knowledge base.** Changing a reference range changes what "correct" means as much as editing a gold label, so `check-data` blocks a run until the manifest is bumped with a changelog note.
- **Rows are validated strictly.** Unknown fields and unknown tags are errors, so a typo can't silently drop a case from an eval.
- **Gold values are worked out by hand from the reference table**, never computed by `normalise.py`. Otherwise MOOD-03 would only be testing the code against itself.
- **Line endings are forced to LF** (`.gitattributes`), and files are hashed after normalising them, so hashes match on Windows and on the Linux CI runner.

## Calibration

- **The labelling sample is every judge failure, then passes spread evenly across judges** (fixed seed).
  *Why:* a random sample would be almost all passes, and agreement on those says little. *Trade-off:* the agreement number is not an unbiased estimate of overall judge accuracy.
- **No invented numbers.** Without human labels the report says "not yet calibrated".
