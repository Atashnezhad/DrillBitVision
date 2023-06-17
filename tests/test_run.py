from pathlib import Path
from neural_network_model.process_data import Preprocessing
from neural_network_model.bit_vision import BitVision
import os


def test_run():

    # check if a dir name test_resouces exists if not create one
    if not os.path.exists(Path(__file__).parent / "test_resources"):
        os.mkdir(Path(__file__).parent / "test_resources")

    # download the images
    obj = Preprocessing(dataset_address=Path(__file__).parent / "test_resources" / "dataset")
    obj.download_images(limit=10)
    print(obj.image_dict)
    obj.augment_data(
        number_of_images_tobe_gen=10,
        augment_data_address=Path(__file__).parent / "test_resources" / "augmented_dataset"
    )
    obj.train_test_split(
        augmented_data_address=Path(__file__).parent / "test_resources" / "augmented_dataset",
        train_test_val_split_dir_address=Path(__file__).parent / "test_resources" / "dataset_train_test_val"
    )

    obj = BitVision(train_test_val_dir=Path(__file__).parent / "test_resources" / "dataset_train_test_val")
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category()
    obj.compile_model()

    model_name = "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{val_accuracy:.2f}_.h5"
    obj.train_model(
        model_save_address=Path(__file__).parent / "test_resources" / "deep_model",
        model_name=model_name,
        epochs=10,
    )
    obj.plot_history(fig_folder_address=Path(__file__).parent / "test_resources" / "figures")

    obj.predict(
        fig_save_address=Path(__file__).parent / "test_resources" / "figures",
        model_path=Path(__file__).parent / "test_resources" / "deep_model" / model_name,
        test_folder_address=Path(__file__).parent / "test_resources" / "dataset_train_test_val" / "test"
    )

    # find list of images in the Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit"
    directory_path = Path(__file__).parent / "test_resources" / "dataset_train_test_val" / "test" / "pdc_bit"
    list_of_images = [str(x) for x in directory_path.glob("*.jpeg")]
    obj.grad_cam_viz(
        model_path=Path(__file__).parent / "test_resources" / "deep_model" / model_name,
        fig_to_save_address=Path(__file__).parent / "test_resources" / "figures",
        img_to_be_applied_path=Path(__file__).parent / "test_resources" / "dataset_train_test_val" / "test" / "pdc_bit" / list_of_images[
            0],
        output_gradcam_fig_name="gradcam.png"
    )
