

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

# get all pip list packages and save to requirements.txt
.PHONY: freeze
freeze:
	@echo "Freezing requirements..."
	@pip freeze > requirements.txt