import pytest

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
