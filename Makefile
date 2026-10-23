PYTHON ?= python
PROMPTS ?= v1

.PHONY: setup test trace check-data eval eval-live record

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# make trace RUN=runs/baseline_v1 TRACE=E2E-002
trace:
	$(PYTHON) -m foxo_evals trace $(RUN) $(TRACE)

check-data:
	$(PYTHON) -m foxo_evals check-data

eval:
	$(PYTHON) -m foxo_evals run --mode replay --prompts $(PROMPTS) --out runs/latest

eval-live:
	$(PYTHON) -m foxo_evals run --mode live --prompts $(PROMPTS) --out runs/live

# Calls the API only for responses that are not recorded yet.
record:
	$(PYTHON) -m foxo_evals run --mode record --prompts $(PROMPTS) --out runs/record
