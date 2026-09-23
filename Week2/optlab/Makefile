PY ?= python
DAY ?= 6

# "make test DAY=3" runs the tests of days 1..3 -> -m "day1 or day2 or day3"
MARKS := $(shell seq 1 $(DAY) | awk '{printf "%sday%s", (NR>1 ? " or " : ""), $$1}')

.PHONY: help install test test-all contracts check mypy lint guard clean

help:
	@echo "make install      install the package and the dev extras"
	@echo "make test DAY=n   run the tests up to day n (default: all)"
	@echo "make test-all     run every test"
	@echo "make contracts    run the interface contract tests only"
	@echo "make check        test-all + mypy --strict + ruff + the no-oracle guard"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -m "$(MARKS)"

test-all:
	$(PY) -m pytest

contracts:
	$(PY) -m pytest -m contract

mypy:
	$(PY) -m mypy --strict

lint:
	$(PY) -m ruff check .

guard:
	$(PY) -m pytest tests/test_no_oracle_in_src.py

check: test-all mypy lint guard

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
