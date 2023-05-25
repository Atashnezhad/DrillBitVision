#SHELL := /bin/bash

# make a parameter to save date and time of execution
DATE := $(shell date +%Y-%m-%d_%H-%M-%S)

# make a new requirements.txt file
.PHONY: new-requirements
new-requirements:
	@touch requirements.txt

# update pip
.PHONY: update-pip
update-pip:
	@echo "Updating pip..."
	@pip install --upgrade pip

# install required packages
.PHONY: install_requirements
install_requirements:
	@make update-pip
	@echo "Installing required packages..."
	@pip install -r requirements.txt
	@pip install -r requirements-test.txt

# get all pip list packages and save to requirements.txt
.PHONY: freeze
freeze:
	@echo "Freezing requirements..."
	@pip freeze > requirements.txt

# deactivate the virtual environment
.PHONY: deactivate
deactivate:
	@echo "Deactivating virtual environment..."
	@deactivate

# remove the folder dataset
#.PHONY: remove-dataset
#remove-dataset:
#	@echo "Removing dataset folder..."
#	@rm -rf dataset

# commit to repository
.PHONY: commit
commit:
	@run-flake8
	@run-black
	@echo "Commiting to repository..."
	@git add .
	@git commit -m "Update: $(DATE)"
	@git push origin master


# run the flake8 script on all python files in the src folder
.PHONY: run-flake8
run-flake8:
	isort neural_network_model/*.py
	#isort test/*.py
	flake8 neural_network_model/*.py
	#flake8 test/*.py

# run black on all python files in the src folder
.PHONY: run-black
run-black:
	black neural_network_model/*.py
	black tests/*.py

# pip freeze
.PHONY: pip-freeze
pip-freeze:
	@pip freeze > requirements.txt

# if a file like .env is not ignored although it should be, run this command
.PHONY: remove-env
remove-env:
	@git rm -r --cached .env

# build tha package
.PHONY: build
build:
	@python3 -m build
	@python setup.py bdist_wheel sdist

# check using twine if the dist/*
.PHONY: check
check:
	@twine check dist/*

# upload to test pypi
.PHONY: upload-test
upload-test:
	@twine upload --repository testpypi dist/*

# upload to pypi
.PHONY: upload
upload:
	@twine upload dist/*

# remove some dir folders
.PHONY: remove
remove:
	@rm -rf dataset dataset_augmented dataset_train_test_val deep_model s3_dataset