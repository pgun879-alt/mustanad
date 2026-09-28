.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup test lint typecheck check demo serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install the package with dev extras
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check formatting and lint rules
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .

typecheck: ## Run mypy
	.venv/bin/mypy

check: lint typecheck test ## Run every check the CI would run

demo: ## Index the sample corpus and ask questions in English and Arabic
	./scripts/demo.sh

serve: ## Run the HTTP API on 127.0.0.1:8000
	$(PYTHON) -m mustanad.cli serve

clean: ## Remove caches and the demo database
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/demo.sqlite3 data/demo.sqlite3-wal data/demo.sqlite3-shm
