import pytest


import sys
from pathlib import Path

# Get the parent directory of the current file (assuming the script is in the test folder)
current_dir = Path(__file__).resolve().parent
# Get the parent directory of the current directory (assuming the test folder is one level below the main folder)
main_dir = current_dir.parent
# Add the main directory to the Python path
sys.path.append(str(main_dir))

from neural_network_model.bit_vision import BitVision


@pytest.fixture
def obj():
    return BitVision()


# skip this test
@pytest.mark.skip(reason="no way of currently testing this")
def test_model_predict_no_model_path(obj):
    with pytest.raises(ValueError):
        obj.predict()
        # assert ValueError model_path is None
        assert ValueError == "model_path is None"
