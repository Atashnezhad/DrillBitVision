import argparse
import os
from pathlib import Path
import sys

# Get the parent directory of the current file (assuming the script is in the test folder)
current_dir = Path(__file__).resolve().parent
# Get the parent directory of the current directory (assuming the test folder is one level below the main folder)
main_dir = current_dir.parent
# Add the main directory to the Python path
sys.path.append(str(main_dir))

import pytest
from neural_network_model.bit_vision import BitVision
from neural_network_model.process_data import Preprocessing


# @pytest.mark.skip(reason="no way of currently testing this")
def test_run():
    # download the images
    obj = Preprocessing(dataset_address=Path(__file__).parent / "dataset")
    obj.download_images(limit=10)
    print(obj.image_dict)
    obj.augment_data(
        number_of_images_tobe_gen=10,
        augment_data_address=Path(__file__).parent / "augmented_dataset"
    )
    obj.train_test_split(
        augmented_data_address=Path(__file__).parent / "augmented_dataset",
        train_test_val_split_dir_address=Path(__file__).parent / "dataset_train_test_val"
    )

    obj = BitVision(train_test_val_dir=Path(__file__).parent / "dataset_train_test_val")
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category()
    obj.compile_model()
    #
    model_name = "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{val_accuracy:.2f}_.h5"
    obj.train_model(
        epochs=8,
        model_save_address=Path(__file__).parent / "deep_model",
        model_name=model_name
    )
    obj.plot_history(fig_folder_address=Path(__file__).parent / "figures")

    best_model = obj.return_best_model_name(directory="deep_model")

    obj.predict(
        fig_save_address=Path(__file__).parent / "figures",
        model_path=Path(__file__).parent / "deep_model" / best_model,
        test_folder_address=Path(__file__).parent / "dataset_train_test_val" / "test"
    )

    # find list of images in the Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit"
    directory_path = Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit"
    list_of_images = [str(x) for x in directory_path.glob("*.jpeg")]

    obj.grad_cam_viz(
        model_path=Path(__file__).parent / "deep_model" / best_model,
        fig_to_save_address=Path(__file__).parent / "figures",
        img_to_be_applied_path=Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit" / list_of_images[
            0],
        output_gradcam_fig_name="gradcam.png"
    )
