.PHONY: release
release:
	rm -rf dist
	scripts/git_tag.sh
	python setup.py sdist bdist_wheel
	twine upload dist/*

.PHONY: docs
docs:
	rm -rf docs/build/html
	@cd docs && sphinx-apidoc -f -e -o source/ ../graphslam/
	@cd docs && make html && make html

.PHONY: test
test:
	python setup.py test

.PHONY: coverage
coverage:
	coverage run --source graphslam setup.py test && coverage html && coverage report -m

.PHONY: tdd
tdd:
	coverage run --source graphslam setup.py test && coverage report -m

.PHONY: lint
lint:
	flake8 graphslam/ && pylint graphslam/ && flake8 tests/ && pylint tests/

.PHONY: alltests
alltests:
	flake8 graphslam/ && pylint graphslam/ && flake8 tests/ && pylint tests/ && coverage run --source graphslam setup.py test && coverage report -m
