import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from tensorflow import keras

# Load environment variables from .env file
load_dotenv()


class DataAddressSetting(BaseModel):
    MAIN_DATA_DIR_ADDRESS: str = Path(__file__).parent / ".." / "dataset"
    TEST_DIR_ADDRESS: str = (
        Path(__file__).parent
        / ".."
        / "dataset_train_test_val"
        / "test"  # check the TRAIN_TEST_SPLIT_DIR_NAMES in
        # the PreprocessingSetting make sure test is in the list
    )


class DownloadImageSetting(BaseModel):
    LIMIT: int = 50


class AugmentationSetting(BaseModel):
    ROTATION_RANGE: int = 25
    WIDTH_SHIFT_RANGE: float = 0.0
    HEIGHT_SHIFT_RANGE: float = 0.0
    SHEAR_RANGE: float = 0.2
    ZOOM_RANGE: float = 0.5
    HORIZONTAL_FLIP: bool = True
    FILL_MODE: str = "nearest"

    BATCH_SIZE: int = 1

    # Address to save augmented images
    AUGMENTED_IMAGES_DIR_ADDRESS: str = (
        Path(__file__).parent / ".." / "dataset_augmented"
    )
    AUGMENTED_IMAGES_SAVE_PREFIX: str = "augmented_image"
    AUGMENTED_IMAGES_SAVE_FORMAT: str = "jpeg"

    NUMBER_OF_IMAGES_TOBE_GENERATED: int = 200


class CategorySetting(BaseModel):
    CATEGORIES: list = ["pdc_bit", "rollercone_bit"]


class IgnoreSetting(BaseModel):
    IGNORE_LIST: list = [".DS_Store", ".pytest_cache", "__pycache__"]


class PreprocessingSetting(BaseModel):
    TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS: str = (
        Path(__file__).parent / ".." / "dataset_train_test_val"
    )
    TRAIN_TEST_SPLIT_DIR_NAMES: list = ["train", "test", "val"]
    TRAIN_FRACTION: float = 0.7
    TEST_FRACTION: float = 0.2
    VAL_FRACTION: float = 0.1


class ModelSetting(BaseModel):
    MODEL_PATH: str = (
        Path(__file__).parent
        / ".."
        / "deep_model"
        / "model_epoch_27_loss_0.52_acc_0.64_val_acc_0.62_.h5"
    )

    # compile
    LOSS: str = "categorical_crossentropy"
    METRICS: List = ["accuracy"]

    # fit generator
    EPOCHS: int = 5
    FIT_GEN_VERBOSE: int = 1
    VALIDATION_STEPS: int = 2
    CLASS_WEIGHT: dict = None
    MAX_QUEUE_SIZE: int = 10
    WORKERS: int = 1
    SHUFFLE: bool = True
    INITIAL_EPOCH: int = 1
    USE_MULTIPROCESSING: bool = False

    # checkpoint
    PERIOD: int = 1
    MONITOR: str = "val_accuracy"
    CHECK_POINT_VERBOSE: int = 1
    SAVE_BEST_ONLY: bool = True
    MODE: str = "max"

    NUMBER_OF_TEST_TO_PRED: int = 35


class FigureSetting(BaseModel):
    # in the predict method of BitVision class
    FIGURE_SIZE_IN_PRED_MODEL: tuple = (20, 15)
    NUM_ROWS_IN_PRED_MODEL: int = 5
    NUM_COLS_IN_PRED_MODEL: int = 7
    FIG_PRED_OUT_DIR_ADDRESS: str = Path(__file__) / ".." / ".." / "figures"

    # in the plot_history method of BitVision class
    FIGURE_SIZE_IN_PLOT_HIST: tuple = (17, 10)
    NUM_ROWS_IN_PLOT_HIST: int = 1
    NUM_COLS_IN_PLOT_HIST: int = 2


class RandomSeedSetting(BaseModel):
    SEED: int = 1


class FlowFromDirectorySetting(BaseModel):
    TARGET_SIZE: tuple = (224, 224)
    COLOR_MODE: str = "rgb"
    CLASS_MODE: str = "categorical"
    BATCH_SIZE: int = 32
    SHUFFLE: bool = True
    SEED: int = 1234


class DataGenSetting(BaseModel):
    RESCALE: float = 1.0 / 255.0


class GradCamSetting(BaseModel):
    IMG_PATH: str = Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_1.png"

    LAST_CONV_LAYER_NAME: str = "conv2d_2"
    IMAGE_NEW_NAME: str = (
        Path(__file__).parent / ".." / "figures" / "grad_cam_pdc_Image_1.png"
    )

    ALPHA: float = 0.7


class S3BucketSetting(BaseModel):
    BUCKET_NAME: str = "bitimages123"
    PARENT_FOLDER_NAME: str = "dataset"
    SUBFOLDER_NAME: list = ["dataset/pdc_bit", "dataset/rollercone_bit"]
    DOWNLOAD_LOCATION: str = (Path(__file__).parent / ".." / "s3_dataset").resolve()
    REGION_NAME: str = "us-east-2"
    # s3 access credentials
    AWS_S3_SECRET_KEY: str = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    AWS_S3_ACCESS_KEY: str = os.getenv("S3_AWS_ACCESS_KEY")


class Ec2Setting(BaseModel):
    ACCESS_KEY: str = os.getenv("EC2_ACCESS_KEY")
    SECRET_KEY: str = os.getenv("EC2_SECRET_KEY")

    AMI_ID: str = "ami-08333bccc35d71140"
    INSTANCE_TYPE: str = "t2.micro"
    REGION_NAME: str = "us-east-2"

    KEY_NAME: str = "bitvision_ec2"
    PEM_FILE_ADDRESS: str = (
        Path(__file__).parent / "ec2_key" / "bitvision_ec2.pem"
    ).resolve()

    SSH_USER: str = "ec2-user"

    # security group
    SECURITY_GROUP_ID: str = os.environ.get("EC2_SECURITY_GROUP_ID")
    SECURITY_GROUP_DESCRIPTION: str = "bitvision_security_group"
    SECURITY_GROUP_NAME: str = "bitvision_security_group"
    VPC_ID: str = "vpc-020d4d0022fbc35b3"


class GitHubSetting(BaseModel):
    REPO_URL: str = "https://github.com/Atashnezhad/DrillBitVision.git"
    BRANCH_NAME: str = "main"

    FOLDER_NAME: str = "myproject"


class Setting(BaseModel):
    AUGMENTATION_SETTING: AugmentationSetting = AugmentationSetting()
    PREPROCESSING_SETTING: PreprocessingSetting = PreprocessingSetting()
    CATEGORY_SETTING: CategorySetting = CategorySetting()
    MODEL_SETTING: ModelSetting = ModelSetting()
    RANDOM_SEED_SETTING: RandomSeedSetting = RandomSeedSetting()
    FLOW_FROM_DIRECTORY_SETTING: FlowFromDirectorySetting = FlowFromDirectorySetting()
    FIGURE_SETTING: FigureSetting = FigureSetting()
    IGNORE_SETTING: IgnoreSetting = IgnoreSetting()
    DATA_ADDRESS_SETTING: DataAddressSetting = DataAddressSetting()
    DATA_GEN_SETTING: DataGenSetting = DataGenSetting()
    DOWNLOAD_IMAGE_SETTING: DownloadImageSetting = DownloadImageSetting()
    GRAD_CAM_SETTING: GradCamSetting = GradCamSetting()
    S3_BUCKET_SETTING: S3BucketSetting = S3BucketSetting()
    EC2_SETTING: Ec2Setting = Ec2Setting()
    GITHUB_SETTING: GitHubSetting = GitHubSetting()


SETTING = Setting()


if __name__ == "__main__":
    print(SETTING.EC2_SETTING.SECURITY_GROUP_ID)
