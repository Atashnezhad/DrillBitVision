from pathlib import Path

import pytest
from neural_network_model.bit_vision import BitVision


@pytest.fixture
def obj():
    return BitVision()


def test_model_predict_no_model_path(obj):
    with pytest.raises(ValueError):
        obj.predict()
        # assert ValueError model_path is None
        assert ValueError == "model_path is None"
