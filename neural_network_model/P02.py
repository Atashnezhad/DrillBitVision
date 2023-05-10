# import pandas as pd
import os
from model import AUGMENTATION_SETTING
import logging

# import shutil
from tqdm import tqdm

# import sys
import warnings
from pathlib import Path
from random import random
from typing import List, Any, Dict

warnings.filterwarnings("ignore")
# from collections import Counter
# import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# from tensorflow import keras
# import tensorflow

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)

# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


class BitVision:
    """
    This class is used to train the neural network model.
    for the bit type detection.
    """

    def __init__(self, *args, **kwargs):
        self.dataset_address = kwargs.get("dataset_address", None)
        self.image_dict: Dict[str, Dict[str, Any]] = {}
        self.categories_name_folders = None
        self.find_categories_name()

    def find_categories_name(self) -> None:
        """
        This function is used to find the categories name.
        :return:
        """
        self.categories_name_folders = os.listdir(self.dataset_address)
        list_to_ignore = [".DS_Store", ".gif"]
        self.categories_name_folders = [
            x for x in self.categories_name_folders if x not in list_to_ignore
        ]
        logger.info(self.categories_name_folders)
        logger.info(f"Categories name: {self.categories_name_folders}")

        # for images in each category dir, check if the image is in list_to_ignore, if yes remove from dir
        for category_folder in self.categories_name_folders:
            # read the files in the dataset folder
            main_dataset_folder = Path(__file__).parent / ".." / "dataset"
            data_address = main_dataset_folder / category_folder

            for file in os.listdir(data_address):
                if file in list_to_ignore:
                    os.remove(file)
                    logger.info(f"Removed {file} from {category_folder}")

    def read_data(self, *, print_file_names: bool = False) -> None:
        """
        This function prints the number of files in the dataset folder.
        :param print_file_names:

        :return:
        """

        for category_folder in self.categories_name_folders:
            # read the files in the dataset folder
            main_dataset_folder = Path(__file__).parent / ".." / "dataset"
            data_address = main_dataset_folder / category_folder

            if print_file_names:
                logger.info(f"List of files in the folder: {category_folder}")
                for file in data_address.iterdir():
                    logger.info(f"File name: {file.name}")

            self.image_dict[category_folder] = {}
            self.image_dict[category_folder]["image_list"] = list(
                data_address.iterdir()
            )
            self.image_dict[category_folder]["number_of_images"] = len(
                list(data_address.iterdir())
            )

        logger.info(f"Image dict: {self.image_dict}")

    def balance_data(self, number_of_images_tobe_gen: int = 200):
        """
        This function is used to balance the data.
        :return:
        """
        Augment_data_gen = image.ImageDataGenerator(
            rotation_range=AUGMENTATION_SETTING.ROTATION_RANGE,
            width_shift_range=AUGMENTATION_SETTING.WIDTH_SHIFT_RANGE,
            height_shift_range=AUGMENTATION_SETTING.HEIGHT_SHIFT_RANGE,
            shear_range=AUGMENTATION_SETTING.SHEAR_RANGE,
            zoom_range=AUGMENTATION_SETTING.ZOOM_RANGE,
            horizontal_flip=AUGMENTATION_SETTING.HORIZONTAL_FLIP,
            fill_mode=AUGMENTATION_SETTING.FILL_MODE,
        )

        for image_category in self.image_dict.keys():
            # check if a dir dataset_augmented exists if not create it
            if not os.path.exists(
                AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category
            ):
                os.makedirs(
                    AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category
                )
            # check if the dir is empty if not delete all the files
            # TODO: check if this is needed or not
            else:
                for file in (
                    AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category
                ).iterdir():
                    os.remove(file)

            number_of_images = self.image_dict[image_category]["number_of_images"]
            for _ in tqdm(
                range(0, number_of_images_tobe_gen + 1),
                desc=f"Augmenting {image_category} images:",
            ):
                # generate a random number integer between 0 and number_of_images
                rand_img_num = int(random() * number_of_images)

                img_address = self.image_dict[image_category]["image_list"][
                    rand_img_num
                ]
                if img_address == ".DS_Store":
                    logger.info(f"Found .DS_Store in {image_category} folder")
                    continue
                logger.info(f"Image address: {img_address}")

                img = load_img(img_address)

                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for _ in Augment_data_gen.flow(
                    x,
                    batch_size=AUGMENTATION_SETTING.BATCH_SIZE,
                    save_to_dir=AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                    / image_category,
                    save_prefix=AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_PREFIX,
                    save_format=AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_FORMAT,
                ):
                    break

    def train_test_split(self, *args, **kwargs):
        """
        This function is used to split the data into train and test.
        :return:
        """
        train_fraction = kwargs.get("train_fraction", 0.7)
        test_fraction = kwargs.get("test_fraction", 0.2)
        validation_fraction = kwargs.get("validation_fraction", 0.1)

        logger.info(f"Train fraction: {train_fraction}")
        logger.info(f"Test fraction: {test_fraction}")
        logger.info(f"Validation fraction: {validation_fraction}")

        # get the list of dirs in the AUGMENTED_IMAGES_DIR_ADDRESS
        augmented_images_dir_list = os.listdir(
            AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )
        logger.info(f"Augmented images dir list: {augmented_images_dir_list}")

        # make a new dir for train and test and validation data
        if not os.path.exists(AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS):
            os.makedirs(AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS)
        # make 3 dirs for train, test and validation under AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
        for dir_name in AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            if not os.path.exists(
                AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS / dir_name
            ):
                os.makedirs(
                    AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS / dir_name
                )
        logger.info(f"Created train, test and validation dirs")

        # under each train, test and validation dir make a dir for each category
        for dir_name in AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            for category in augmented_images_dir_list:
                if not os.path.exists(
                    AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                    / dir_name
                    / category
                ):
                    os.makedirs(
                        AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                        / dir_name
                        / category
                    )
        logger.info(f"Created category dirs under train, test and validation dirs")







if __name__ == "__main__":
    obj = BitVision(dataset_address=Path(__file__).parent / ".." / "dataset")
    # obj.read_data(print_file_names=False)
    # obj.balance_data(number_of_images_tobe_gen=200)

    train_test_validation_split = {
        "train_fraction": 0.7,
        "test_fraction": 0.2,
        "validation_fraction": 0.1,
    }
    obj.train_test_split(**train_test_validation_split)
