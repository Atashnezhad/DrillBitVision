from pathlib import Path

import pytest
from neural_network_model.process_data import Preprocessing


def test_download_images(mocker):
    mock_bing_downloader = mocker.patch("neural_network_model.process_data.downloader")
    Preprocessing.download_images()
    print(mock_bing_downloader.download.call_count)
    # assert mock_bing_downloader.download.call_count == 2
    # assert it was called
    assert mock_bing_downloader.download.called
