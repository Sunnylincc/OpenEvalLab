.PHONY: install test demo clean

install:
	python -m pip install -e .

test:
	pytest

demo:
	openevallab demo

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf results reports
