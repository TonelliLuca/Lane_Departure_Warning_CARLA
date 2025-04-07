.PHONY: clean virtualenv lint requirements

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python

virtualenv: ## Create virtualenv - activate with source env/bin/activate
	virtualenv -p $(PYTHON_INTERPRETER) env

requirements: ## Install Python dependencies
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint: ## Lint using flake8
	flake8 src

