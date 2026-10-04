.PHONY: help install test lint format typecheck ask validate check

PYTHON ?= python
Q ?= How fast does a falling object hit the ground if it is dropped from 20 m?

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov --cov-report=term-missing

lint:  ## Check lint and formatting (no changes)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-format and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:  ## Run mypy in strict mode
	$(PYTHON) -m mypy

ask:  ## Ask a question with the fake LLM: make ask Q="..."
	askphysics ask "$(Q)"

validate:  ## Validate all seed data
	askphysics validate-data

check: lint typecheck test  ## Everything CI runs
