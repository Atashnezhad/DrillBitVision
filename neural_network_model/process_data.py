import logging
import os
import random
import shutil

from bing_image_downloader import downloader

from neural_network_model.model import SETTING
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

# set seed to get the same random numbers each time
random.seed(1)
import random

# import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

# import shutil
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.FATAL)
# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.FATAL)


# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


class Preprocessing:
    """
    This class is used to train the neural network model.
    for the bit type detection.
    """

    def __init__(self, *args, **kwargs):
        self.dataset_address = kwargs.get("dataset_address", None)
        self.image_dict: Dict[str, Dict[str, Any]] = {}
        self.categories_name_folders = None
        self.find_categories_name()

    @staticmethod
    def download_images(category_list=None) -> None:
        if category_list is None:
            category_list = SETTING.CATEGORY_SETTING.CATEGORIES
        for category in category_list:
            downloader.download(
                category,
                limit=200,
                output_dir=Path(__file__).parent / ".." / "dataset",
                adult_filter_off=True,
                force_replace=False,
                timeout=120,
            )
        logger.info("Downloaded images")

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

    def make_image_dict(self, *, print_file_names: bool = False) -> None:
        """
        This function reads the data from the dataset folder.
        and stores the data in image_dict.
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

    def augment_data(self, number_of_images_tobe_gen: int = 200):
        """
        This function augments the images and save them into the dataset_augmented folder.
        :param number_of_images_tobe_gen: number of images to be generated
        :return: None
        """
        Augment_data_gen = image.ImageDataGenerator(
            rotation_range=SETTING.AUGMENTATION_SETTING.ROTATION_RANGE,
            width_shift_range=SETTING.AUGMENTATION_SETTING.WIDTH_SHIFT_RANGE,
            height_shift_range=SETTING.AUGMENTATION_SETTING.HEIGHT_SHIFT_RANGE,
            shear_range=SETTING.AUGMENTATION_SETTING.SHEAR_RANGE,
            zoom_range=SETTING.AUGMENTATION_SETTING.ZOOM_RANGE,
            horizontal_flip=SETTING.AUGMENTATION_SETTING.HORIZONTAL_FLIP,
            fill_mode=SETTING.AUGMENTATION_SETTING.FILL_MODE,
        )

        for image_category in self.image_dict.keys():
            # check if a dir dataset_augmented exists if not create it
            if not os.path.exists(
                SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                / image_category
            ):
                os.makedirs(
                    SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                    / image_category
                )
            # check if the dir is empty if not delete all the files
            # TODO: check if this is needed or not
            else:
                for file in (
                    SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                    / image_category
                ).iterdir():
                    os.remove(file)

            number_of_images = self.image_dict[image_category]["number_of_images"]
            for _ in tqdm(
                range(0, number_of_images_tobe_gen + 1),
                desc=f"Augmenting {image_category} images:",
            ):
                # generate a random number integer between 0 and number_of_images
                rand_img_num = int(random.random() * number_of_images)

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
                    batch_size=SETTING.AUGMENTATION_SETTING.BATCH_SIZE,
                    save_to_dir=SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                    / image_category,
                    save_prefix=SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_PREFIX,
                    save_format=SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_FORMAT,
                ):
                    break

    def train_test_split(self, *args, **kwargs):
        """
        This function is used to split the data into train, test and validation.
        And save them into the dataset_train_test_split folder.
        :return:
        """
        # get the list of dirs in the AUGMENTED_IMAGES_DIR_ADDRESS
        augmented_images_dir_list = os.listdir(
            SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )
        logger.info(f"Augmented images dir list: {augmented_images_dir_list}")

        # make a new dir for train and test and validation data
        if not os.path.exists(
            SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
        ):
            os.makedirs(SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS)
        # make 3 dirs for train, test and validation under AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
        for dir_name in SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            if not os.path.exists(
                SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS / dir_name
            ):
                os.makedirs(
                    SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                    / dir_name
                )
        logger.info(f"Created train, test and validation dirs")

        # under each train, test and validation dir make a dir for each category
        for dir_name in SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            for category in augmented_images_dir_list:
                if not os.path.exists(
                    SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                    / dir_name
                    / category
                ):
                    os.makedirs(
                        SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                        / dir_name
                        / category
                    )
        logger.info(f"Created category dirs under train, test and validation dirs")

        self.populated_augmented_images_into_train_test_val_dirs()

    def populated_augmented_images_into_train_test_val_dirs(self):
        for category in self.categories_name_folders:
            # get the list of images in the dataset_augmented pdc_bit folder
            original_list = os.listdir(
                SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS / category
            )

            # Number of items to select for each new list
            num_train = int(
                len(original_list) * SETTING.PREPROCESSING_SETTING.TRAIN_FRACTION
            )
            num_test = int(
                len(original_list) * SETTING.PREPROCESSING_SETTING.TEST_FRACTION
            )
            num_val = len(original_list) - num_train - num_test

            # Randomly select items from the original list without replacement
            selected_items = random.sample(original_list, len(original_list))

            # Create three new lists containing 70%, 20%, and 10% of the original list
            train_list = selected_items[:num_train]
            test_list = selected_items[num_train : num_train + num_test]
            val_list = selected_items[num_train + num_test :]
            # TODO: check if the dirs are not empty if not delete all the files
            # copy the images from the dataset_augmented pdc_bit folder to the train, test and validation folders
            self.copy_images(train_list, category, "train")
            self.copy_images(test_list, category, "test")
            self.copy_images(val_list, category, "val")

    @staticmethod
    def copy_images(images, categ, dest_folder):
        for image in images:
            shutil.copy(
                SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                / categ
                / image,
                SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
                / dest_folder
                / categ
                / image,
            )
        logger.info(
            f"copied {len(images)} images for category {categ} to {dest_folder}"
        )


if __name__ == "__main__":
    obj = Preprocessing(dataset_address=Path(__file__).parent / ".." / "dataset")
    # BitVision.download_images()
    obj.make_image_dict(print_file_names=False)
    obj.augment_data(number_of_images_tobe_gen=200)
    obj.train_test_split()
