# import pandas as pd
import logging
import os
import random
import shutil

from keras import Sequential
from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)

from neural_network_model.model import SETTING

# set seed to get the same random numbers each time
random.seed(1)
import matplotlib.pyplot as plt

# import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
from collections import Counter
import random

import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
from tensorflow import keras
import tensorflow


warnings.filterwarnings("ignore")
# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# from tensorflow import keras
# import tensorflow


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
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
    This class is used to train a neural network model for bit vision.
    """

    def __init__(self, *args, **kwargs):
        self.train_test_val_dir: str = kwargs.get(
            "train_test_val_dir",
            SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS,
        )

    @property
    def categories(self) -> List[str]:
        """
        This function returns the list of categories.
        :return: list of categories
        """
        # get the list of dirs in the resource_dir
        subdir_name = os.listdir(self.train_test_val_dir)
        subdir_path = self.train_test_val_dir / subdir_name[0]
        categories_name = os.listdir(subdir_path)
        return categories_name

    @property
    def data_details(self) -> int:
        resource_dir = Path(__file__).parent / ".." / self.train_test_val_dir
        # get the list of dirs in the resource_dir
        subdir_name = os.listdir(resource_dir)
        data = {}
        # for each subdir in the resource_dir, get the list of files
        for subdir in subdir_name:
            data[subdir] = {}
            subdir_path = resource_dir / subdir
            for category in os.listdir(subdir_path):
                category_path = subdir_path / category
                # for each file in the category, find the number of files
                num_files = len(os.listdir(category_path))
                # logger.info(f"Number of files in {category_path.resolve()} is {num_files}")
                data[subdir][category] = num_files
        return data

    def deep_net_model(self):
        model = Sequential()
        activ = "relu"

        model.add(
            Conv2D(32, kernel_size=(3, 3), activation=activ, input_shape=(224, 224, 3))
        )
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(3, 3), activation=activ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(3, 3), activation=activ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64, activation=activ))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation="softmax"))

    def plot_image_category(self, *args, **kwargs):
        """
        This function is used to plot images.
        :param images: list of images
        :param labels: list of labels
        :return:
        """
        nrows = kwargs.get("nrows", 1)
        number_of_categories = len(self.categories)
        ncols = kwargs.get("ncols", number_of_categories)
        fig_size = kwargs.get("fig_size", (17, 10))

        # get one image for each category in train data and plot them
        fig, axs = plt.subplots(nrows, ncols, figsize=fig_size)
        for category in self.categories:
            category_path = self.train_test_val_dir / "train" / category
            image_path = category_path / os.listdir(category_path)[1]
            img = load_img(image_path)
            axs[self.categories.index(category)].imshow(img)
            axs[self.categories.index(category)].set_title(category)
        plt.show()


if __name__ == "__main__":
    obj = BitVision()
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category()
