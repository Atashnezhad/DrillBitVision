from pathlib import Path
from unittest import mock

import pytest
from neural_network_model.process_data import Preprocessing


def test_download_images(mocker):
    #
    mock_bing_downloader = mocker.patch("neural_network_model.process_data.downloader")
    Preprocessing.download_images()
    print(mock_bing_downloader.download.call_count)
    # assert mock_bing_downloader.download.call_count == 2
    # assert it was called
    assert mock_bing_downloader.download.called


def myfunc(*args, **kwargs):
    print(kwargs)
    return None


# TODO: work on this tests
def test_augment_data():
    # Testing code with the `image_dict` mocked
    with mock.patch.object(Preprocessing, "image_dict") as mocked_image_dict:
        mocked_image_dict.return_value = {
            "category1": {
                "image_list": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                "number_of_images": 2,
            },
            "category2": {
                "image_list": ["/path/to/image3.jpg", "/path/to/image4.jpg"],
                "number_of_images": 2,
            },
        }

        # Test code that uses the mocked image_dict
        preprocessing = Preprocessing()
        image_dict = preprocessing.image_dict
        print(image_dict)


def my_dict():
    return {
        "category1": {
            "image_list": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
            "number_of_images": 2,
        },
        "category2": {
            "image_list": ["/path/to/image3.jpg", "/path/to/image4.jpg"],
            "number_of_images": 2,
        },
    }


def test_augment_data_2():
    # Create the patch object
    mocked_image_dict = mock.patch.object(Preprocessing, "image_dict")

    # Start patching the image_dict property
    mocked_image_dict.start()
    mocked_image_dict.return_value = {
        "category1": {
            "image_list": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
            "number_of_images": 2,
        },
        "category2": {
            "image_list": ["/path/to/image3.jpg", "/path/to/image4.jpg"],
            "number_of_images": 2,
        },
    }
    # Test code that uses the mocked image_dict
    preprocessing = Preprocessing()
    image_dict = preprocessing.image_dict
    print(image_dict.return_value)

    # Stop patching the image_dict property
    mocked_image_dict.stop()


import unittest

from unittest.mock import patch


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessing = Preprocessing()
        self.image_dict_mock = mock.Mock()
        self.image_dict_mock.return_value = {
            "category1": ["image1", "image2"],
            "category2": ["image3", "image4"],
        }

        with patch.object(self.preprocessing, "image_dict", self.image_dict_mock):
            self.test_image_dict()

    def test_image_dict(self):
        self.assertEqual(
            self.preprocessing.image_dict,
            {"category1": ["image1", "image2"], "category2": ["image3", "image4"]},
        )
