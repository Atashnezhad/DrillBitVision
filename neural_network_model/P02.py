# import pandas as pd
import os
# import shutil
# import sys
import warnings
from pathlib import Path
from random import random
from typing import List, Any, Dict

warnings.filterwarnings('ignore')
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


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
        self.categories_name_folders = [x for x in self.categories_name_folders if x not in list_to_ignore]
        # print(self.categories_name_folders)

    def read_data(self, *, print_file_names: bool = False) -> None:
        """
        This function prints the number of files in the dataset folder.
        :param print_file_names:
        :param folders_name: list of folders name
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

            self.image_dict[category_folder]["image_list"] = list(data_address.iterdir())
            self.image_dict[category_folder]["number_of_images"] = len(list(data_address.iterdir()))

        print(self.image_dict)

    def balance_data(self, number_of_images_tobe_gen: int = 200):
        """
        This function is used to balance the data.
        :return:
        """
        Augment_data_gen = image.ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.2,
            zoom_range=0.5,
            horizontal_flip=True,
            fill_mode='nearest')

        for image_category in self.image_dict.keys():
            for i in range(0, number_of_images_tobe_gen):

                # select a random img
                rand_img_num = random.randint(0, NORMAL_count - 1)
                img_name = NORMAL_img_list[rand_img_num]

                img_address = NORMAL_dir + img_name
                img = load_img(img_address)

                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for batch in Augment_data_gen. \
                        flow(x, batch_size=1,
                             save_to_dir='../Dataset_augmented/NORMAL/',
                             save_prefix='augmented_normal',
                             save_format='jpeg'):
                    break


if __name__ == "__main__":
    obj = BitVision(dataset_address="../dataset")
    obj.read_data(print_file_names=False)
