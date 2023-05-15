from pathlib import Path
from typing import List

from tensorflow import keras
from pydantic import BaseModel


class DataAddressSetting(BaseModel):
    MAIN_DATA_DIR_ADDRESS: str = Path(__file__).parent / ".." / "dataset"
    TEST_DIR_ADDRESS: str = Path(__file__).parent / ".." / "dataset_train_test_val" / "test"


class DownloadImageSettig(BaseModel):
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
    IGNORE_LIST: list = [".DS_Store"]


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
        Path(__file__).parent / ".." / "deep_model" / "model_epoch_39_loss_0.28_acc_0.79_val_acc_0.66_.h5"
    )

    # compile
    LOSS: str = "categorical_crossentropy"
    METRICS: List = ["accuracy"]

    # fit generator
    EPOCHS: int = 3
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
    FIG_PRED_OUT_DIR_ADDRESS: str = (Path(__file__) / ".." / ".." / "figures")

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
    DOWNLOAD_IMAGE_SETTING: DownloadImageSettig = DownloadImageSettig()


SETTING = Setting()
