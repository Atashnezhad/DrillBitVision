from pathlib import Path

from neural_network_model.bit_vision import BitVision
from neural_network_model.process_data import Preprocessing


def main():
    parent_dir = Path(__file__).resolve().parent.parent
    # download the images
    obj = Preprocessing(dataset_address=parent_dir / "dataset")
    obj.download_images(limit=8)
    print(obj.image_dict)
    obj.augment_data(
        number_of_images_tobe_gen=10,
        augment_data_address=parent_dir / "dataset_augmented",
    )
    obj._train_test_split(
        augmented_data_address=parent_dir / "dataset_augmented",
        train_test_val_split_dir_address=parent_dir / "dataset_train_test_val",
    )

    obj = BitVision(train_test_val_dir=parent_dir / "dataset_train_test_val")
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category()
    obj.compile_model()
    #
    model_name = "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{val_accuracy:.2f}_.h5"
    obj.train_model(
        epochs=8, model_save_address=parent_dir / "deep_model", model_name=model_name
    )
    obj.plot_history(fig_folder_address=parent_dir / "figures")

    best_model = obj.return_best_model_name(directory=parent_dir / "deep_model")

    obj.predict(
        fig_save_address=parent_dir / "figures",
        model_path=parent_dir / "deep_model" / best_model,
        test_folder_address=parent_dir / "dataset_train_test_val" / "test",
    )

    # find list of images in the parent_dir / "dataset_train_test_val" / "test" / "pdc_bit"
    directory_path = parent_dir / "dataset_train_test_val" / "test" / "pdc_bit"
    list_of_images = [str(x) for x in directory_path.glob("*.jpeg")]

    obj.grad_cam_viz(
        model_path=parent_dir / "deep_model" / best_model,
        fig_to_save_address=parent_dir / "figures",
        img_to_be_applied_path=parent_dir
        / "dataset_train_test_val"
        / "test"
        / "pdc_bit"
        / list_of_images[0],
        output_gradcam_fig_name="test.png",
    )


if __name__ == "__main__":
    main()
