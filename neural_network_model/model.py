from pathlib import Path
from typing import List

from tensorflow import keras
from pydantic import BaseModel


class AugmentionSetting(BaseModel):
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


class CategorySetting(BaseModel):
    CATEGORIES: list = ["pdc_bit", "rollercone_bit"]


class PreprocessingSetting(BaseModel):
    TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS: str = (
            Path(__file__).parent / ".." / "dataset_train_test_val"
    )
    TRAIN_TEST_SPLIT_DIR_NAMES: list = ["train", "test", "val"]
    TRAIN_FRACTION: float = 0.7
    TEST_FRACTION: float = 0.2
    VAL_FRACTION: float = 0.1


class ModelSetting(BaseModel):
    MODELS_DIR_ADDRESS: str = (
            Path(__file__).parent / ".." / "deep_model"
    )
    SAVE_FILE_PATH: str = MODELS_DIR_ADDRESS / "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{" \
                                               "val_accuracy:.2f}_.h5"

    # compile
    LOSS: str = "categorical_crossentropy"
    METRICS: List = ["accuracy"]

    # fit generator
    EPOCHS: int = 40
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


class RandomSeedSetting(BaseModel):
    SEED: int = 1


class FlowFromDirectorySetting(BaseModel):
    TARGET_SIZE: tuple = (224, 224)
    COLOR_MODE: str = "rgb"
    CLASS_MODE: str = "categorical"
    BATCH_SIZE: int = 32
    SHUFFLE: bool = True
    SEED: int = 1234


class Setting(BaseModel):
    AUGMENTATION_SETTING: AugmentionSetting = AugmentionSetting()
    PREPROCESSING_SETTING: PreprocessingSetting = PreprocessingSetting()
    CATEGORY_SETTING: CategorySetting = CategorySetting()
    MODEL_SETTING: ModelSetting = ModelSetting()
    RANDOM_SEED_SETTING: RandomSeedSetting = RandomSeedSetting()
    FLOW_FROM_DIRECTORY_SETTING: FlowFromDirectorySetting = FlowFromDirectorySetting()


SETTING = Setting()
