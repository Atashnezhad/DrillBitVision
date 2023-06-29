# ignore the warning
import warnings
from pathlib import Path

import pytest

from neural_network_model.process_data import Preprocessing
from neural_network_model.transfer_learning import TransferModel

warnings.filterwarnings("ignore")


@pytest.mark.skip(reason="skip the test")
def test_run():
    # download the dataset
    obj = Preprocessing()
    obj.download_images(limit=30)

    transfer_model = TransferModel(
        dataset_address=Path(__file__).parent / ".." / "dataset"
    )
    transfer_model.plot_data_images(num_rows=3, num_cols=3)
    transfer_model.train_model()
    transfer_model.plot_metrics_results()
    transfer_model.results()
    transfer_model.predcit_test()
    transfer_model.grad_cam_viz(num_rows=3, num_cols=2)

    assert True
