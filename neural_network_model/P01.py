from pathlib import Path


def main(*, folder_name: str = "pdc_bit", print_file_names: bool = False) -> None:
    """
    This function prints the number of files in the dataset folder.
    :param print_file_names:
    :param folder_name:
    :return:
    """
    # read the files in the dataset folder
    main_dataset_folder = Path(__file__).parent / ".." / "dataset"
    data_address = main_dataset_folder / folder_name

    if print_file_names:
        print("List of files in the folder:")
        for file in data_address.iterdir():
            print(file.name)

    # print number of files in the folder
    print("Number of files in the folder:")
    print(len(list(data_address.iterdir())))


if __name__ == "__main__":
    main(folder_name="pdc_bit", print_file_names=True)
