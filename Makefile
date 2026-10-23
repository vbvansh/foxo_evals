PYTHON ?= python

.PHONY: setup test trace check-data

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# make trace RUN=runs/baseline_v1 TRACE=E2E-002
trace:
	$(PYTHON) -m foxo_evals trace $(RUN) $(TRACE)

check-data:
	$(PYTHON) -m foxo_evals check-data
