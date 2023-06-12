from pathlib import Path, PosixPath
from typing import List, Union
from unittest import mock
from bing_image_downloader import downloader

import pytest

from neural_network_model.model import SETTING
from neural_network_model.process_data import Preprocessing
from neural_network_model.s3 import MyS3


@pytest.fixture
def _object():
    return Preprocessing()


def test_download_images_1(mocker, _object):
    #
    mock_bing_downloader = mocker.patch("neural_network_model.process_data.downloader")
    _object.download_images()
    print(mock_bing_downloader.download.call_count)
    assert mock_bing_downloader.download.call_count == 2


# note here the mock is used which is from unittest while in
# test_download_images_2, the mocker is used which is from pytest
def test_download_images_2(_object):
    with mock.patch(
            "neural_network_model.process_data.downloader"
    ) as mock_bing_downloader:
        _object.download_images()
        print(mock_bing_downloader.download.call_count)
        assert mock_bing_downloader.download.call_count == 2


def test_download_images_3(mocker, _object):
    mock_bing_downloader_download = mocker.patch(
        "neural_network_model.process_data.downloader.download"
    )
    mock_logger_info = mocker.patch("neural_network_model.process_data.logger.info")
    _object.download_images()
    print(mock_bing_downloader_download.call_count)
    assert (
            mock_bing_downloader_download.call_count == 2
    )  # by default, the number of categories to download is 2
    assert mock_logger_info.call_count == 1
    # assert mock_logger_info.call_args_list[0][0][0] == "Downloaded images"
    assert mock_logger_info.call_args_list[0].args[0] == "Downloaded images"


def test_download_images_s3_1(mocker, _object):
    mock_my_s3_obj = mocker.patch.object(MyS3, "download_files_from_subfolders")
    _object.download_images(from_s3=True)
    assert mock_my_s3_obj.call_count == 1


def side_effect_test_download_images_s3_2(*args, **kwargs) -> None:
    print(args)


def test_download_images_s3_2(mocker, _object):
    mock_download_files = mocker.patch("neural_network_model.process_data.MyS3.download_files_from_subfolders",
                                       side_effect=side_effect_test_download_images_s3_2)
    _object.download_images(from_s3=True)
    mock_download_files.assert_called_once_with('bitimages123',
                                                ['dataset/pdc_bit', 'dataset/rollercone_bit'],
                                                PosixPath('/Users/amin/Downloads/DrillBitVision/s3_dataset')
                                                )


def side_effect_test_property_1(*args, **kwargs) -> Union[List[str], List[PosixPath]]:
    first_os_arge = (Path(__file__).parent / ".." / "dataset").resolve()
    print(args[0], "-----")
    if args[0] == first_os_arge:
        return ['pdc_bit', 'rollercone_bit']
    else:
        return ['test_preprocessing.py', 'test_bitvision.py']


def test_property_1(mocker, _object):
    # set dataset_address
    _object.dataset_address = Path(__file__).parent / ".." / "dataset"

    # mock the os listdir function
    mock_os_listdir = mocker.patch(
        "neural_network_model.process_data.os.listdir",
        side_effect=side_effect_test_property_1,
    )

    assert _object.categorie_name == [
        'test_preprocessing.py',
        'test_bitvision.py'
    ]
