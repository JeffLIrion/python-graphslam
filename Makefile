.PHONY: docs
docs:
	rm -rf docs/build/html
	@cd docs && sphinx-apidoc -f -e -o source/ ../graphslam/
	@cd docs && make html && make html

.PHONY: doxygen
doxygen:
	rm -rf docs/html
	doxygen Doxyfile

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
	flake8 graphslam/ && pylint graphslam/

.PHONY: alltests
alltests:
	flake8 graphslam/ && pylint graphslam/ && coverage run --source graphslam setup.py test && coverage report -m
