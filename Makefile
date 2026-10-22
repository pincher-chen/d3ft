PYTHON ?= python
.PHONY: check test figures case-list
check:
	$(PYTHON) scripts/verify_response_release.py
test:
	$(PYTHON) scripts/check_response_cpu.py
figures:
	$(PYTHON) scripts/response_figures.py --band-python "$(or $(BAND_PYTHON),$(PYTHON))"
case-list:
	$(PYTHON) scripts/run_response.py --list
