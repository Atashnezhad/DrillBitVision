import argparse
import os
import sys
from pathlib import Path

# Get the parent directory of the current file (assuming the script is in the test folder)
current_dir = Path(__file__).resolve().parent
# Get the parent directory of the current directory (assuming the test folder is one level below the main folder)
main_dir = current_dir.parent
# Add the main directory to the Python path
sys.path.append(str(main_dir))

from neural_network_model.bit_vision import BitVision  # noqa: E402
from neural_network_model.process_data import Preprocessing  # noqa: E402


def get_parser():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Example Argument Parser")

    # Add arguments
    parser.add_argument("--dataset_address", help="Path to the dataset directory")
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit for downloading images"
    )
    parser.add_argument(
        "--augmented_data_address", help="Path to the augmented dataset directory"
    )
    parser.add_argument(
        "--train_test_val_split_dir_address",
        help="Path to the train/test/val split directory",
    )
    parser.add_argument(
        "--train_test_val_dir", help="Path to the train/test/val directory"
    )
    parser.add_argument("--model_save_address", help="Path to save the trained model")
    parser.add_argument("--model_name", help="Name of the trained model file")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument("--fig_save_address", help="Path to save the figures")
    parser.add_argument("--model_path", help="Path to the trained model")
    parser.add_argument("--test_folder_address", help="Path to the test folder")
    parser.add_argument(
        "--img_to_be_applied_path", help="Path to the image for Grad-CAM visualization"
    )
    parser.add_argument(
        "--output_gradcam_fig_name",
        default="gradcam.png",
        help="Name of the output Grad-CAM figure",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    # Get the arguments
    args = get_parser()
    # Access the values of the arguments
    dataset_address = Path(args.dataset_address) if args.dataset_address else None
    limit = args.limit
    augmented_data_address = (
        Path(args.augmented_data_address) if args.augmented_data_address else None
    )
    train_test_val_split_dir_address = (
        Path(args.train_test_val_split_dir_address)
        if args.train_test_val_split_dir_address
        else None
    )
    train_test_val_dir = (
        Path(args.train_test_val_dir) if args.train_test_val_dir else None
    )
    model_save_address = (
        Path(args.model_save_address) if args.model_save_address else None
    )
    model_name = args.model_name
    # epochs = args.epochs
    fig_save_address = Path(args.fig_save_address) if args.fig_save_address else None
    # model_path = Path(args.model_path) if args.model_path else None
    test_folder_address = (
        Path(args.test_folder_address) if args.test_folder_address else None
    )
    # img_to_be_applied_path = (
    #     Path(args.img_to_be_applied_path) if args.img_to_be_applied_path else None
    # )
    output_gradcam_fig_name = args.output_gradcam_fig_name

    # Perform actions based on the arguments
    if not dataset_address:
        dataset_address = Path(__file__).parent / "test_resources" / "dataset"
    if not augmented_data_address:
        augmented_data_address = (
            Path(__file__).parent / "test_resources" / "augmented_dataset"
        )
    if not train_test_val_split_dir_address:
        train_test_val_split_dir_address = (
            Path(__file__).parent / "test_resources" / "dataset_train_test_val"
        )
    if not train_test_val_dir:
        train_test_val_dir = (
            Path(__file__).parent / "test_resources" / "dataset_train_test_val"
        )
    if not model_save_address:
        model_save_address = Path(__file__).parent / "test_resources" / "deep_model"
    if not model_name:
        model_name = "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{val_accuracy:.2f}_.h5"
    if not fig_save_address:
        fig_save_address = Path(__file__).parent / "test_resources" / "figures"
    if not test_folder_address:
        test_folder_address = Path(__file__).parent / "test_resources" / "test"

    # check if a dir name test_resouces exists if not create one
    if not os.path.exists(Path(__file__).parent / "test_resources"):
        os.mkdir(Path(__file__).parent / "test_resources")

    # download the images
    obj = Preprocessing(dataset_address=dataset_address)
    obj.download_images(limit=limit)
    print(obj.image_dict)
    obj.augment_data(
        number_of_images_tobe_gen=10, augment_data_address=augmented_data_address
    )
    obj._train_test_split(
        augmented_data_address=augmented_data_address,
        train_test_val_split_dir_address=train_test_val_split_dir_address,
    )

    obj = BitVision(train_test_val_dir=train_test_val_dir)
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category()
    obj.compile_model()

    model_name = "model_epoch_{epoch:02d}_loss_{loss:.2f}_acc_{accuracy:.2f}_val_acc_{val_accuracy:.2f}_.h5"
    obj.train_model(
        model_save_address=model_save_address,
        model_name=model_name,
        epochs=10,
    )
    obj.plot_history(fig_folder_address=fig_save_address)

    obj.predict(
        fig_save_address=fig_save_address,
        model_path=model_save_address / model_name,
        test_folder_address=train_test_val_dir / "test",
    )

    # find list of images in the Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit"
    directory_path = train_test_val_dir / "test" / "pdc_bit"
    list_of_images = [str(x) for x in directory_path.glob("*.jpeg")]
    obj.grad_cam_viz(
        model_path=model_save_address / model_name,
        fig_to_save_address=fig_save_address,
        img_to_be_applied_path=train_test_val_dir
        / "test"
        / "pdc_bit"
        / list_of_images[0],
        output_gradcam_fig_name=output_gradcam_fig_name,
    )


if __name__ == "__main__":
    main()
