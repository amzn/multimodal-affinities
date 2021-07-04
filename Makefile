.PHONY: install

install:
	python -m pip install --upgrade pip
	pip install -e .

clean:
	find . -name '__pycache__' | xargs -I rm
	find . -name '*.pyc' | xargs -I rm

