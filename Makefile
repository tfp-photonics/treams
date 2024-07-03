clean:
	find . -type f -name '*.so' -delete
	find . -type f -name '*.c' -delete

build:
	rm -rf dist
	pip install build
	python -m build