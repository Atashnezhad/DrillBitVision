import logging
import os
import random
import shutil
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from bing_image_downloader import downloader
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm

from neural_network_model.model import SETTING
from neural_network_model.s3 import MyS3

# Set seed to get the same random numbers each time
random.seed(SETTING.RANDOM_SEED_SETTING.SEED)

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

logging.basicConfig(level=logging.FATAL)


class Preprocessing:
    """
    This class is used to train the neural network model.
    for the bit type detection.
    """

    def __init__(self, *args, **kwargs):
        """
        This function is used to initialize the class.
        :param args:
        :param kwargs:
        """
        self.dataset_address = kwargs.get("dataset_address", None)
        self.categories_name_folders = None
        self.sub_catego_data_address = None

    def download_images(self, category_list=None, from_s3=False, limit=None) -> None:
        """
        This function is used to download the images from the internet.
        :param category_list:
        :param from_s3: bool
        :param limit: int
        :param download_location_address:
        :return:
        """
        if from_s3:
            # download the images from the S3 bucket
            s3 = MyS3()

            # Specify your bucket name and folders
            bucket_name = SETTING.S3_BUCKET_SETTING.BUCKET_NAME
            subfolders = SETTING.S3_BUCKET_SETTING.SUBFOLDER_NAME
            download_location_address = (
                self.dataset_address or SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION
            )

            s3.download_files_from_subfolders(
                bucket_name, subfolders, download_location_address
            )
            logger.debug("Downloaded images from S3 bucket")
            return

        if category_list is None:
            category_list = SETTING.CATEGORY_SETTING.CATEGORIES
        for category in category_list:
            downloader.download(
                category,
                limit=limit or SETTING.DOWNLOAD_IMAGE_SETTING.LIMIT,
                output_dir=self.dataset_address
                or SETTING.DATA_ADDRESS_SETTING.MAIN_DATA_DIR_ADDRESS,
                adult_filter_off=True,
                force_replace=False,
                timeout=120,
            )
        logger.debug("Downloaded images")

    @property
    def categorie_name(self) -> List[str]:
        """
        This function is used to find the categories name.
        :return:
        """
        self.categories_name_folders = os.listdir(self.dataset_address)
        list_to_ignore = SETTING.IGNORE_SETTING.IGNORE_LIST
        self.categories_name_folders = [
            x for x in self.categories_name_folders if x not in list_to_ignore
        ]
        logger.debug(self.categories_name_folders)
        logger.debug(f"Categories name: {self.categories_name_folders}")

        # for images in each category dir, check if the image is in list_to_ignore, if yes remove from dir
        for category_folder in self.categories_name_folders:
            # read the files in the dataset folder
            main_dataset_folder = (
                self.dataset_address
                or SETTING.DATA_ADDRESS_SETTING.MAIN_DATA_DIR_ADDRESS
            )
            data_address = main_dataset_folder / category_folder

            for file in os.listdir(data_address):
                if file in list_to_ignore:
                    os.remove(data_address / file)
                    logger.debug(f"Removed {file} from {category_folder}")

        return self.categories_name_folders

    @property
    def image_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        This function reads the data from the dataset folder.
        and stores the data as a dict with categories and values in image_dict.
        :return:
        """
        image_dicts = {}
        for category_folder in self.categorie_name:
            # read the files in the dataset folder
            main_dataset_folder = (
                self.dataset_address
                or SETTING.DATA_ADDRESS_SETTING.MAIN_DATA_DIR_ADDRESS
            )
            # check if the main_datast_folder is not None if yes then raise error
            if main_dataset_folder is None:
                logging.error("main_dataset_folder is None")
                raise ValueError("main_dataset_folder is None")

            self.sub_catego_data_address = main_dataset_folder / category_folder
            image_dicts[category_folder] = {}
            image_dicts[category_folder]["image_list"] = list(
                self.sub_catego_data_address.iterdir()
            )
            image_dicts[category_folder]["number_of_images"] = len(
                list(self.sub_catego_data_address.iterdir())
            )

        return image_dicts

    def augment_data(
        self,
        number_of_images_tobe_gen: int = SETTING.AUGMENTATION_SETTING.NUMBER_OF_IMAGES_TOBE_GENERATED,
        augment_data_address: Path = SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS,
    ):
        """
        This function augments the images and save them into the dataset_augmented folder.
        :param number_of_images_tobe_gen: number of images to be generated
        :param augment_data_address: address to save the augmented images
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

        main_address = (
            augment_data_address
            or SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )

        for image_category in self.image_dict.keys():
            # check if a dir dataset_augmented exists if not create it
            if not os.path.exists(main_address / image_category):
                os.makedirs(main_address / image_category)
            # check if the dir is empty if not delete all the files
            else:
                # empty the previous augmented images
                for file in (main_address / image_category).iterdir():
                    os.remove(file)

            number_of_images = self.image_dict[image_category]["number_of_images"]
            for _ in tqdm(
                range(0, number_of_images_tobe_gen),
                desc=f"Augmenting {image_category} images:",
            ):
                # generate a random number integer between 0 and number_of_images
                rand_img_num = int(random.random() * number_of_images)

                img_address = self.image_dict[image_category]["image_list"][
                    rand_img_num
                ]
                for case in SETTING.IGNORE_SETTING.IGNORE_LIST:
                    if case == img_address.name:
                        logger.debug(f"Found {case} in {image_category} folder")
                        continue
                # check if the image name is in the ignore list, if so continue
                if img_address in SETTING.IGNORE_SETTING.IGNORE_LIST:
                    continue

                logger.debug(f"Image address: {img_address}")

                img = load_img(img_address)

                x = img_to_array(img)
                x = x.reshape((1,) + x.shape())

                for _ in Augment_data_gen.flow(
                    x,
                    batch_size=SETTING.AUGMENTATION_SETTING.BATCH_SIZE,
                    save_to_dir=main_address / image_category
                    or SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
                    / image_category,
                    save_prefix=SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_PREFIX,
                    save_format=SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_SAVE_FORMAT,
                ):
                    break

    def _train_test_split(self, *args, **kwargs):
        """
        This function is used to split the data into train, test and validation.
        And save them into the dataset_train_test_split folder.
        :param args:
        :param kwargs: augmented_data_address, train_test_val_split_dir_address
        :return:
        """
        augmented_data_address = (
            kwargs.get("augmented_data_address")
            or SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )
        train_test_val_split_dir_address = (
            kwargs.get("train_test_val_split_dir_address")
            or SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS
        )
        # get the list of dirs in the AUGMENTED_IMAGES_DIR_ADDRESS
        augmented_images_dir_list = os.listdir(augmented_data_address)
        logger.debug(f"Augmented images dir list: {augmented_images_dir_list}")

        # make a new dir for train and test and validation data
        if not os.path.exists(train_test_val_split_dir_address):
            os.makedirs(train_test_val_split_dir_address)
        # make 3 dirs for train, test and validation under AUGMENTATION_SETTING.TRAIN_TEST_SPLIT_DIR_ADDRESS
        for dir_name in SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            if not os.path.exists(train_test_val_split_dir_address / dir_name):
                os.makedirs(train_test_val_split_dir_address / dir_name)
        logger.debug("Created train, test and validation dirs")

        # under each train, test and validation dir make a dir for each category
        for dir_name in SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES:
            for category in augmented_images_dir_list:
                if not os.path.exists(
                    train_test_val_split_dir_address / dir_name / category
                ):
                    os.makedirs(train_test_val_split_dir_address / dir_name / category)
        logger.debug("Created category dirs under train, test and validation dirs")

        self._populated_augmented_images_into_train_test_val_dirs(
            augmented_data_address, train_test_val_split_dir_address
        )

    def _populated_augmented_images_into_train_test_val_dirs(self, *arges, **kwargs):
        dataset_augmented_dir_address = arges[0]
        train_test_val_split_dir_address = arges[1]

        for category in self.categories_name_folders:
            # get the list of images in the dataset_augmented category (i.e. pdc_bit) folder
            original_list = os.listdir(dataset_augmented_dir_address / category)

            # Number of items to select for each new list
            num_train = int(
                len(original_list) * SETTING.PREPROCESSING_SETTING.TRAIN_FRACTION
            )
            num_test = int(
                len(original_list) * SETTING.PREPROCESSING_SETTING.TEST_FRACTION
            )

            # Randomly select items from the original list without replacement
            selected_items = random.sample(original_list, len(original_list))

            # Create three new lists containing 70%, 20%, and 10% of the original list
            train_list = selected_items[:num_train]
            test_list = selected_items[num_train : num_train + num_test]  # noqa: E203
            val_list = selected_items[num_train + num_test :]  # noqa: E203
            # copy the images from the dataset_augmented pdc_bit folder to the train, test and validation folders
            self._copy_images(
                train_list,
                category,
                "train",
                dataset_augmented_dir_address,
                train_test_val_split_dir_address,
            )
            self._copy_images(
                test_list,
                category,
                "test",
                dataset_augmented_dir_address,
                train_test_val_split_dir_address,
            )
            self._copy_images(
                val_list,
                category,
                "val",
                dataset_augmented_dir_address,
                train_test_val_split_dir_address,
            )

    @staticmethod
    def _copy_images(
        images,
        categ,
        dest_folder,
        dataset_augmented_dir_address,
        train_test_val_split_dir_address,
    ):
        train_test_val_split_dir_address = (
            train_test_val_split_dir_address
            or SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS
        )
        dataset_augmented_dir_address = (
            dataset_augmented_dir_address
            or SETTING.AUGMENTATION_SETTING.AUGMENTED_IMAGES_DIR_ADDRESS
        )

        # check if the dir is empty, if not delete all the files
        if os.listdir(train_test_val_split_dir_address / dest_folder / categ):
            logger.debug(
                f"Deleting all the files in {train_test_val_split_dir_address / dest_folder / categ}"
            )
            for file in os.listdir(
                train_test_val_split_dir_address / dest_folder / categ
            ):
                os.remove(train_test_val_split_dir_address / dest_folder / categ / file)

        for _image in images:
            shutil.copy(
                dataset_augmented_dir_address / categ / _image,
                train_test_val_split_dir_address / dest_folder / categ / _image,
            )
        logger.debug(
            f"copied {len(images)} images for category {categ} to {dest_folder}"
        )

    @staticmethod
    def print_deepnet_project_metadata():
        return SETTING.load_settings_from_json()


if __name__ == "__main__":
    obj = Preprocessing(dataset_address=Path(__file__).parent / ".." / "dataset")
    obj.download_images()

    # or download the data from s3. this is after you have downloaded the data
    # using Process.download_images() and uploaded it to s3
    # this way the data would be consistent across all the users
    # obj = Preprocessing(dataset_address=Path(__file__).parent / ".." / "s3_dataset")
    # obj.download_images(from_s3=True)

    print(obj.image_dict)
    obj.augment_data(number_of_images_tobe_gen=10)
    obj._train_test_split()
    obj.print_deepnet_project_metadata()
