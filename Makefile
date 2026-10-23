PYTHON ?= python
PROMPTS ?= v1
RUN ?= runs/baseline_v1
A ?= runs/baseline_v1
B ?= runs/doctor_v2

.PHONY: setup test trace check-data eval eval-live record baseline compare regression-demo calibration-template calibrate

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

calibration-template:
	$(PYTHON) -m foxo_evals calibration-template --run $(RUN)

calibrate:
	$(PYTHON) -m foxo_evals calibrate

baseline:
	$(PYTHON) -m foxo_evals run --mode replay --prompts v1 --out runs/baseline_v1

compare:
	$(PYTHON) -m foxo_evals compare $(A) $(B)

# Replays Doctor v1 and v2 from recordings and shows what the gate catches. The v2 run
# and the compare are expected to fail, so make ignores their exit codes ("-").
regression-demo:
	$(PYTHON) -m foxo_evals run --mode replay --prompts v1 --out runs/demo_v1
	-$(PYTHON) -m foxo_evals run --mode replay --prompts v1,doctor=v2 --out runs/demo_v2
	-$(PYTHON) -m foxo_evals compare runs/demo_v1 runs/demo_v2
