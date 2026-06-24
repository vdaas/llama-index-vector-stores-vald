PYTHON ?= python3

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_/-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Format with ruff.
	ruff format llama_index tests
	ruff check --fix llama_index tests

lint:	## Lint with ruff and mypy.
	ruff format --check llama_index tests
	ruff check llama_index tests
	mypy llama_index

test:	## Run pytest.
	pytest tests

version/python:	## Print the Python version used for CI (consumed by vald-client-ci setup-python).
	@echo 3.10

ci/deps/install:	## Install build dependencies for the release pipeline.
	$(PYTHON) -m pip install --upgrade pip build

ci/package/prepare:	## Build the sdist and wheel into dist/ (hatchling via PEP 517).
	$(PYTHON) -m build
