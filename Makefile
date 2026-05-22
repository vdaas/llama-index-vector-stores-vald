help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Format with ruff.
	ruff format llama_index tests
	ruff check --fix llama_index tests

lint:	## Lint with ruff and mypy.
	ruff format --check llama_index tests
	ruff check llama_index tests
	mypy llama_index

test:	## Run pytest.
	pytest tests
