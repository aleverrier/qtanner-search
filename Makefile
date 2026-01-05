SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help setup smoke test

help:
	@echo "Targets:"
	@echo "  make setup  - install this package into the active Python environment (editable)"
	@echo "  make smoke  - fast syntax/import checks (compileall)"
	@echo "  make test   - run the repo quick checks via ./scripts/run_tests.sh"

setup:
	python -m pip install -e .

smoke:
	python -m compileall -q .

test:
	@if [ ! -f "./scripts/run_tests.sh" ]; then echo "ERROR: ./scripts/run_tests.sh not found"; exit 1; fi
	@if [ ! -x "./scripts/run_tests.sh" ]; then echo "ERROR: ./scripts/run_tests.sh exists but is not executable"; echo "Fix: chmod +x scripts/run_tests.sh"; exit 1; fi
	./scripts/run_tests.sh
