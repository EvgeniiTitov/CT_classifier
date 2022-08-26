install-requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

install-pre-commit:
	pre-commit install

pre-commit:
	pre-commit run -a -v

test:
	pytest tests -v

clean:
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete