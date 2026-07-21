.PHONY: install install-dev run train test test-fast test-api lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python3 run.py

train:
	python3 ml_training/train.py --quick --epochs 2 --no-fine-tune

test:
	python3 -m pytest tests/ -v

test-fast:
	python3 -m pytest tests/ -v -m "not slow"

test-api:
	python3 -m pytest tests/test_api.py -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
