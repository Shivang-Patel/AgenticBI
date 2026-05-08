.PHONY: install test lint run clean help

PYTHON := python3
PIP := pip3

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

test: ## Run pytest suite
	pytest -v --tb=short

lint: ## Run black and flake8
	black app/ tests/
	flake8 app/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format: ## Auto-format code with black and isort
	black app/ tests/
	isort app/ tests/

run: ## Run the AgenticBI dashboard
	$(PYTHON) app/main.py

clean: ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ dist/ build/

check: ## Run all checks (lint + test)
	make lint
	make test

requirements: ## Export pinned requirements
	$(PIP) freeze > requirements.txt
