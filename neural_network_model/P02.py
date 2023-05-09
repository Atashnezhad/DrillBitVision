# import pandas as pd
import os
# import shutil
# import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings('ignore')
# from collections import Counter
# import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# from tensorflow import keras
# import tensorflow
#
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class BitVision:
    """
    This class is used to train the neural network model.
    for the bit type detection.
    """
    def __init_(self, *args, **kwargs):
        pass

    def read_data(self, *, folders_name: List[str] = None,
                  print_file_names: bool = False) -> None:
        """
        This function prints the number of files in the dataset folder.
        :param print_file_names:
        :param folders_name: list of folders name
        :return:
        """
        my_dict = {}

        for folder_name in folders_name:
            # read the files in the dataset folder
            main_dataset_folder = Path(__file__).parent / ".." / "dataset"
            data_address = main_dataset_folder / folder_name

            if print_file_names:
                print("List of files in the folder:")
                for file in data_address.iterdir():
                    print(file.name)

            # print number of files in the folder
            # print(f"Number of files in the folder {folder_name}:")
            # print(len(list(data_address.iterdir())))

            my_dict[folder_name] = len(list(data_address.iterdir()))

        print(my_dict)


if __name__ == "__main__":
    obj = BitVision()
    obj.read_data(folders_name=["pdc_bit", "rollercone_bit"], print_file_names=False)
