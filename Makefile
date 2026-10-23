PYTHON ?= python

.PHONY: setup test

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest
