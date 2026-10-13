PYTHON ?= python
.PHONY: check test figures case-list
check:
	$(PYTHON) scripts/verify_release.py
test:
	$(PYTHON) scripts/run_tests.py --suite cpu
figures:
	$(PYTHON) make_figures.py
case-list:
	$(PYTHON) scripts/run_case.py Al --dry-run
