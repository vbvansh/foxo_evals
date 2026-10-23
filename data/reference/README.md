# Reference data

**Illustrative only — not clinical guidance.** Reference ranges, critical thresholds,
unit conversion factors, drug/lab interactions and food tags are simplified so the
demo has realistic structure. Do not use them for real patients.

| File | Used by | What it holds |
|---|---|---|
| `biomarkers.json` | Mood (normalisation), evals | canonical names, aliases, unit factors, sex-specific ranges, critical thresholds |
| `interactions.jsonl` | Doctor, DOC-07 | medication + abnormal biomarker → warning |
| `food_lexicon.json` | COACH-01 | food → tags, and which tags each diet, allergy and condition forbids |
