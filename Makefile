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
	#black test/*.py