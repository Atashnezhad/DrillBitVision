import sys
import warnings
from pathlib import Path, PosixPath
from typing import List, Union
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# Get the parent directory of the current file (assuming the script is in the test folder)
current_dir = Path(__file__).resolve().parent
# Get the parent directory of the current directory (assuming the test folder is one level below the main folder)
main_dir = current_dir.parent
# Add the main directory to the Python path
sys.path.append(str(main_dir))


import neural_network_model  # noqa: E402
from neural_network_model.model import SETTING  # noqa: E402
from neural_network_model.process_data import Preprocessing  # noqa: E402
from neural_network_model.s3 import MyS3  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


# skip this test
@pytest.mark.skip
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
    mock_download_files = mocker.patch(
        "neural_network_model.process_data.MyS3.download_files_from_subfolders",
        side_effect=side_effect_test_download_images_s3_2,
    )
    _object.download_images(from_s3=True)

    bucket_name = SETTING.S3_BUCKET_SETTING.BUCKET_NAME
    subfolders = SETTING.S3_BUCKET_SETTING.SUBFOLDER_NAME
    download_location_address = SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION

    mock_download_files.assert_called_once_with(
        bucket_name, subfolders, download_location_address
    )


def side_effect_test_property_1(*args, **kwargs) -> Union[List[str], List[PosixPath]]:
    first_os_arge = (Path(__file__).parent / ".." / "dataset").resolve()
    print(args[0], "-----")
    if args == first_os_arge:
        print("first_os_arge")
        return ["pdc_bit", "rollercone_bit"]
    else:
        print("second_os_arge")
        return ["test_preprocessing.py", "test_bitvision.py"]


# skip this test TODO: fix this test later
@pytest.mark.skip
def test_property_1(mocker, _object):
    # set dataset_address
    _object.dataset_address = Path(__file__).parent / ".." / "dataset"

    # mock the os listdir function
    mocker.patch(
        "neural_network_model.process_data.os.listdir",
        side_effect=side_effect_test_property_1,
    )
    assert _object.categorie_name == ["test_preprocessing.py", "test_bitvision.py"]


def side_effect_test_property_2(*args, **kwargs) -> Union[List[str], List[PosixPath]]:
    if args[0] == (Path(__file__).parent / ".." / "dataset").resolve():
        return ["pdc_bit", "rollercone_bit"]

    else:
        return ["dummy_file1.txt", "dummy_file2.txt"]


def test_property_2(mocker, _object):
    with patch("os.listdir") as mock_os_listdir:
        mock_os_listdir.side_effect = MagicMock(side_effect=side_effect_test_property_2)
        _object.dataset_address = (Path(__file__).parent / ".." / "dataset").resolve()
        assert _object.categorie_name == ["pdc_bit", "rollercone_bit"]


# skip this test TODO: fix this test later
@pytest.mark.skip
def test_property_image_dict(mocker, _object):
    # assign a dummy dataset address
    _object.dataset_address = (Path(__file__).parent / "dummy_dataset").resolve()

    mocker.patch(
        "neural_network_model.process_data.Preprocessing.categorie_name",
        new_callable=mocker.PropertyMock,
        return_value=["pdc_bit", "rollercone_bit"],
    )

    print(_object.image_dict)


@mock.patch.object(
    neural_network_model.process_data.Preprocessing,
    "categorie_name",
    new_callable=mock.PropertyMock,
)
@mock.patch.object(Path, "iterdir")
def test_property_image_dict_2(mock_iterdir, mock_categorie_property, _object):
    # assign a dummy dataset address
    _object.dataset_address = (Path(__file__).parent / "dummy_dataset").resolve()

    mock_categorie_property.return_value = ["pdc_bit", "rollercone_bit"]

    assert _object.image_dict == {
        "pdc_bit": {"image_list": [], "number_of_images": 0},
        "rollercone_bit": {"image_list": [], "number_of_images": 0},
    }


@mock.patch.object(
    neural_network_model.process_data.Preprocessing,
    "categorie_name",
    new_callable=mock.PropertyMock,
)
@mock.patch.object(Path, "iterdir")
def test_property_image_dict_3(mock_iterdir, mock_categorie_property, _object):
    # assign a dummy dataset address
    _object.dataset_address = (Path(__file__).parent / "dummy_dataset").resolve()
    mock_categorie_property.return_value = ["pdc_bit", "rollercone_bit"]

    assert _object.image_dict == {
        "pdc_bit": {"image_list": [], "number_of_images": 0},
        "rollercone_bit": {"image_list": [], "number_of_images": 0},
    }


def test_integrated(_object):
    _object = Preprocessing(dataset_address=Path(__file__).parent / ".." / "dataset")
    _object.download_images()
    # _object.augment_data(number_of_images_tobe_gen=10)
    # _object.train_test_split()


class ImageObject:
    def ImageDataGenerator(self, *args, **kwargs):
        # print(args, kwargs)
        return self

    def flow(self, *args, **kwargs):
        # print(args, kwargs)
        return []


class XObjClass:
    @staticmethod
    def shape():
        return (1, 2, 3)

    def reshape(self, *args, **kwargs):
        # print(args, kwargs)
        return self


def img_to_array_func(*args, **kwargs):
    # print(args, kwargs)
    return XObjClass


def load_image_func(*args, **kwargs):
    # print(args, kwargs)
    return None


class ImageAddressObject:
    @property
    def name(self):
        return "test_image.jpg"


def image_dict_object(*args, **kwargs):
    # print(args, kwargs)
    return {
        "pdc_bit": {"image_list": [ImageAddressObject], "number_of_images": 0},
        "rollercone_bit": {"image_list": [ImageAddressObject], "number_of_images": 0},
    }


def test_augment_data(mocker, _object):
    # mocker patch the property categories_name
    mocker.patch(
        "neural_network_model.process_data.Preprocessing.categorie_name",
        new_callable=mocker.PropertyMock,
        return_value=["pdc_bit", "rollercone_bit"],
    )

    # mocker patch the image dict object
    mocker.patch(
        "neural_network_model.process_data.Preprocessing.image_dict",
        new_callable=mocker.PropertyMock,
        side_effect=image_dict_object,
    )

    # mocker patch load_image function
    mocker.patch(
        "neural_network_model.process_data.load_img",
        side_effect=load_image_func,
    )

    # mocker patch the img_to_array function
    mocker.patch(
        "neural_network_model.process_data.img_to_array",
        side_effect=img_to_array_func,
    )

    # mocker patch the image module
    mocker.patch(
        "neural_network_model.process_data.image",
        side_effect=ImageObject,
    )

    _object.augment_data(number_of_images_tobe_gen=5)
