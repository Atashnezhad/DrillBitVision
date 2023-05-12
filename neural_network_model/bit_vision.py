# import pandas as pd
import logging
import os
import random
import shutil

import pandas as pd
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
# from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
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
            SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS,
        )
        self.model: tensorflow.keras.models.Sequential = None
        self.assemble_deep_net_model()
        self.train_vali_gens = {}

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
    def data_details(self) -> Dict[str, Dict[str, int]]:
        """
        This property check the dirs and make a dict and save the
        train test val data details in the dict.
        """
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

    def assemble_deep_net_model(self) -> tensorflow.keras.models.Sequential:
        """
        This function is used to assemble the deep net model.
        """
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
        model.add(Dense(len(self.categories), activation="softmax"))
        self.model = model
        return self.model

    def compile_model(self, *args, **kwargs) -> None:
        """
        This function is used to compile the model.
        :return:
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=SETTING.MODEL_SETTING.LOSS,
            metrics=SETTING.MODEL_SETTING.METRICS,
        )

        self.model.summary()

    def plot_image_category(self, *args, **kwargs) -> None:
        """
        This function is used to plot images.
        :param images: list of images
        :param labels: list of labels
        :return:
        """
        nrows = kwargs.get("nrows", 1)
        number_of_categories = len(self.categories)
        ncols = kwargs.get("ncols", number_of_categories)
        subdir = kwargs.get("subdir", "train")
        fig_size = kwargs.get("fig_size", (17, 10))

        # get one image for each category in train data and plot them
        fig, axs = plt.subplots(nrows, ncols, figsize=fig_size)
        for category in self.categories:
            category_path = self.train_test_val_dir / subdir / category
            image_path = category_path / os.listdir(category_path)[1]
            img = load_img(image_path)
            axs[self.categories.index(category)].imshow(img)
            axs[self.categories.index(category)].set_title(category)
        plt.show()

    def rescaling(self):
        """
        This function is used to rescale the images.
        The images were augmented before in the preprocessing step.
        Here we just rescale them.
        """
        my_list = ["train", "val"]
        for subdir in my_list:
            datagen = image.ImageDataGenerator(rescale=1. / 255)

            generator = datagen.flow_from_directory(
                directory=SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS / subdir,
                target_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE,
                color_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.COLOR_MODE,
                classes=None,
                class_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.CLASS_MODE,
                batch_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.BATCH_SIZE,
                shuffle=True,
                seed=SETTING.RANDOM_SEED_SETTING.SEED,
                save_to_dir=None,
                save_prefix="",
                save_format="png",
                follow_links=False,
                subset=None,
                interpolation="nearest"
            )
            self.train_vali_gens[subdir] = generator

            logger.info(f"Rescaling {subdir} data, {generator.class_indices}:")

    def check_points(self) -> ModelCheckpoint:
        check_points = ModelCheckpoint(
            SETTING.MODEL_SETTING.SAVE_FILE_PATH,
            monitor=SETTING.MODEL_SETTING.MONITOR,
            verbose=SETTING.MODEL_SETTING.CHECK_POINT_VERBOSE,
            save_best_only=SETTING.MODEL_SETTING.SAVE_BEST_ONLY,
            mode=SETTING.MODEL_SETTING.MODE,
            # period=SETTING.MODEL_SETTING.PERIOD,
        )
        return check_points

    def train_model(self):
        model_history = self.model.fit_generator(
            generator=self.train_vali_gens["train"],
            epochs=SETTING.MODEL_SETTING.EPOCHS,
            verbose=SETTING.MODEL_SETTING.FIT_GEN_VERBOSE,
            validation_data=self.train_vali_gens["val"],
            validation_steps=SETTING.MODEL_SETTING.VALIDATION_STEPS,
            class_weight=SETTING.MODEL_SETTING.CLASS_WEIGHT,
            max_queue_size=SETTING.MODEL_SETTING.MAX_QUEUE_SIZE,
            workers=SETTING.MODEL_SETTING.WORKERS,
            use_multiprocessing=SETTING.MODEL_SETTING.USE_MULTIPROCESSING,
            shuffle=SETTING.MODEL_SETTING.SHUFFLE,
            initial_epoch=SETTING.MODEL_SETTING.INITIAL_EPOCH,
            callbacks=[self.check_points()]
        )


if __name__ == "__main__":
    obj = BitVision()
    # print(obj.categories)
    print(obj.data_details)
    # obj.plot_image_category()
    obj.compile_model()
    obj.rescaling()
    obj.train_model()
