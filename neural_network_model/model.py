from pathlib import Path

from pydantic import BaseModel


class AugmentationSetting(BaseModel):
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


AUGMENTATION_SETTING = AugmentationSetting()
