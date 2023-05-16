#!/bin/bash

# Clone the GitHub repository
git clone https://github.com/Atashnezhad/DrillBitVision.git

# Change directory to the repository folder
cd DrillBitVision

# Install the requirements
pip install -r requirements.txt

# Run the modules
python process_data.py
python bit_vision.py

# Add more module calls as needed

# End of script
