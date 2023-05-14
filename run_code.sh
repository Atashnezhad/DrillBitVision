#!/bin/bash


pip install -r DrillBitVision/requirements.txt

ls DrillBitVision/
# !rm -rf DrillBitVision/dataset

python3 DrillBitVision/neural_network_model/process_data.py
python3 DrillBitVision/neural_network_model/bit_vision.py