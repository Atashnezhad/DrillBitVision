from pathlib import Path

from pydantic import BaseModel


class AugmentionSetting(BaseModel):
    ROTATION_RANGE: int = 65
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
    TRAIN_TEST_SPLIT_DIR_ADDRESS: str = (
        Path(__file__).parent / ".." / "dataset_train_test_val"
    )
    TRAIN_TEST_SPLIT_DIR_NAMES: list = ["train", "test", "val"]
    TRAIN_FRACTION: float = 0.7
    TEST_FRACTION: float = 0.2
    VAL_FRACTION: float = 0.1


class Setting(BaseModel):
    AUGMENTATION_SETTING: AugmentionSetting = AugmentionSetting()
    PREPROCESSING_SETTING: PreprocessingSetting = PreprocessingSetting()
    CATEGORY_SETTING: CategorySetting = CategorySetting()


SETTING = Setting()
