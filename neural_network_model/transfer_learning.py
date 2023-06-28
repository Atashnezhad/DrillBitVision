import logging
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from neural_network_model.bit_vision import BitVision
from neural_network_model.process_data import Preprocessing


class TransferModel(Preprocessing, BitVision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_df: pd.DataFrame = None

        self._prepare_data()

    def _prepare_data(self):

        image_dir = self.dataset_address
        # Get filepaths and labels
        filepaths = list(image_dir.glob(r'**/*.png'))
        # add those with jpg extension
        # filepaths.extend(list(image_dir.glob(r'**/*.jpg')))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # Concatenate filepaths and labels
        image_df = pd.concat([filepaths, labels], axis=1)

        # Shuffle the DataFrame and reset index
        image_df = image_df.sample(frac=1).reset_index(drop=True)

        self.image_df = image_df
        # Show the result
        print(self.image_df.head(3))
        # log the data was prepared
        logging.info(f"Data was prepared")

    def plot_some_images(self):
        # Display some pictures of the dataset with their labels
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10),
                                 subplot_kw={'xticks': [], 'yticks': []})

        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(self.image_df.Filepath[i]))
            ax.set_title(self.image_df.Label[i])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    transfer_model = TransferModel(dataset_address=Path(__file__).parent / ".." / "dataset")
    transfer_model.plot_some_images()
