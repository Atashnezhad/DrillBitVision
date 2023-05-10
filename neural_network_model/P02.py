# import pandas as pd
import os
from neural_network_model.model import AUGMENTATION_SETTING

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
        list_to_ignore = [".DS_Store"]
        self.categories_name_folders = [
            x for x in self.categories_name_folders if x not in list_to_ignore
        ]
        # print(self.categories_name_folders)

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
                print("List of files in the folder:")
                for file in data_address.iterdir():
                    print(file.name)

            self.image_dict[category_folder] = {}
            self.image_dict[category_folder]["image_list"] = list(
                data_address.iterdir()
            )
            self.image_dict[category_folder]["number_of_images"] = len(
                list(data_address.iterdir())
            )

        print(self.image_dict)

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
                os.mkdir(
                    AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category
                )
            # check if the dir is empty if not delete all the files
            # TODO: check if this is needed or not
            else:
                for file in (
                    AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category
                ).iterdir():
                    os.remove(file)

            for _ in tqdm(
                range(0, number_of_images_tobe_gen + 1),
                desc=f"Augmenting {image_category} images:",
            ):
                number_of_images = self.image_dict[image_category]["number_of_images"]
                # generate a random number integer between 0 and number_of_images
                rand_img_num = int(random() * number_of_images)

                img_address = self.image_dict[image_category]["image_list"][
                    rand_img_num
                ]
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

        # check the number of images in the augmented folder
        for image_category in self.image_dict.keys():
            print(
                f"Number of images in {image_category} folder: "
                f"{len(list((AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / image_category).iterdir()))}"
            )

    def train_test_split(self):
        """
        This function is used to split the data into train and test.
        :return:
        """
        # get the list of dirs in the AUGMENTED_IMAGES_DIR_ADDRESS
        augmented_images_dir_list = os.listdir(
            AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )
        print(augmented_images_dir_list)


if __name__ == "__main__":
    obj = BitVision(dataset_address=Path(__file__).parent / ".." / "dataset")
    # obj.read_data(print_file_names=False)
    # obj.balance_data(number_of_images_tobe_gen=200)
    obj.train_test_split()
