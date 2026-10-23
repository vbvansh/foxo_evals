PYTHON ?= python

.PHONY: setup test trace

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# make trace RUN=runs/baseline_v1 TRACE=E2E-02
trace:
	$(PYTHON) -m foxo_evals trace $(RUN) $(TRACE)
