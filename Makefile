# Makefile for gender bias research pipeline

.PHONY: help install dev-install lint format test clean run-pipeline

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with uv
	uv pip install -e .

dev-install:  ## Install development dependencies
	uv pip install -e ".[dev,notebooks]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check .

format:  ## Format code with ruff
	ruff format .

lint-fix:  ## Run ruff with auto-fix
	ruff check . --fix

type-check:  ## Run mypy type checking
	mypy src/

test:  ## Run tests with pytest
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

setup-data-dirs:  ## Create data directories
	mkdir -p data/{raw,processed,embeddings,models}

# Pipeline commands
label-data:  ## Run Stage 1: Gender labeling
	python scripts/01_label_data.py --input data/raw/EJR_gender_dataset_Jan2018.csv --output data/processed/labeled_posts.csv --validate

scrub-text:  ## Run Stage 2: Text scrubbing  
	python scripts/02_scrub_text.py --input data/processed/labeled_posts.csv --output data/processed/scrubbed_posts.csv --analyze --validate

generate-embeddings:  ## Run Stage 3: Generate embeddings
	python scripts/03_generate_embeddings.py --input data/processed/scrubbed_posts.csv --methods bert_cls bert_mean openai_ada --output-dir data/embeddings/

train-models:  ## Run Stage 4: Train models
	python scripts/04_train_models.py --embedding-dir data/embeddings/ --output-dir data/models/ --compare-methods

run-pipeline:  ## Run complete pipeline
	$(MAKE) label-data
	$(MAKE) scrub-text  
	$(MAKE) generate-embeddings
	$(MAKE) train-models