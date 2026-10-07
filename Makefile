.PHONY: help install test lint format typecheck ask validate check screenshots brand site site-smoke setup update conflicts corpus train eval release models

PYTHON ?= python
Q ?= How fast does a falling object hit the ground if it is dropped from 20 m?
MODEL ?= fermi-solem-1

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

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

screenshots:  ## Regenerate docs/images from real CLI output (needs Node + Playwright)
	$(PYTHON) scripts/screenshots.py

brand:  ## Render the logo, banner, and model artwork (needs Pillow, Node + Playwright)
	$(PYTHON) scripts/brand.py

site:  ## Build the askphysics.vercel.app site into build/site
	sh scripts/build_site.sh

site-smoke: site  ## Ask questions through the built site in headless Chromium (needs Playwright)
	node scripts/site_smoke.mjs

# ---------------------------------------------------------------- workflows (scripts/*.sh)

setup:  ## First-time setup: .venv, install, check the data
	sh scripts/setup.sh

update:  ## Pull main and reinstall after a merge
	sh scripts/update.sh

conflicts:  ## Test-merge against origin/main without changing anything
	sh scripts/check.sh --fast --conflicts

corpus:  ## Build the 20M-word prose corpus (Mac kept awake)
	sh scripts/corpus.sh

train:  ## Back up, train, and score a model: make train MODEL=fermi-solem-1
	sh scripts/train.sh $(MODEL)

eval:  ## Score a model on 3,000 held-out questions: make eval MODEL=fermi-tellus-1
	sh scripts/eval.sh $(MODEL)

models:  ## List installed models and backups
	sh scripts/models.sh list

release:  ## Package installed models for a weights release: make release TAG=models-v0.4.0
	sh scripts/release_weights.sh $(TAG)
