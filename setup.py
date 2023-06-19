#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# with open("README.md") as f:
#     long_description = f.read()

long_description = """
The code is for image preprocessing for a neural network. 
It contains functions to download images from Bing, read data from a directory, 
and augment the images. The code also includes a class called Preprocessing. 
The class has methods to download images, find categories, get image data, and 
augment the images.

The Preprocessing class has an initializer that takes an argument dataset_address 
and assigns it to an instance variable dataset_address. The download_images method 
downloads images from the internet using the bing_image_downloader library. 
The categories_name method reads the dataset directory and returns a list of categories. 
It also removes any files that are in the ignore list specified in the SETTING module. 
The image_dict method returns a dictionary with the number of images and a list of image 
file paths for each category. The augment_data method augments the images in each 
category using the ImageDataGenerator class from keras.preprocessing.image module. 
The augmented images are saved to the dataset_augmented directory.

### Bit Vision Module
The BitVision class is designed to train a neural network model for bit vision. It provides various methods for assembling and training the model, as well as performing predictions and visualizations.
The class has an initialization method that sets the train_test_val_dir attribute, which represents the directory where the training, testing, and validation data is stored. It also initializes other attributes such as model, train_vali_gens, and model_class_indices.
The categories property returns a list of categories based on the subdirectories in the train_test_val_dir. It filters out any unwanted categories specified in the IGNORE_LIST.
The data_details property returns a dictionary containing the details of the training, testing, and validation data. It counts the number of files in each category and stores the information in the dictionary.
The assemble_deep_net_model method creates a sequential model for the neural network. It adds convolutional, batch normalization, max pooling, dropout, and dense layers to the model based on predefined settings.
The compile_model method compiles the model by specifying the optimizer, loss function, and metrics to be used during training.
The plot_image_category method plots images from the training data for each category. It accepts parameters such as the number of rows and columns in the plot, the subdirectory (e.g., "train"), and the figure size.
The _rescaling method is responsible for rescaling the images before training. It uses the ImageDataGenerator class from Keras to rescale the images based on the specified settings.
The check_points method returns a ModelCheckpoint object, which is used to save the best model during training based on a specified monitor metric.
The train_model method trains the model using the training and validation data. It uses the fit_generator function to perform the training, specifying various parameters such as the number of epochs, validation data, class weights, and callbacks (including the checkpoint).
The plot_history method plots the training and validation loss and accuracy over epochs. It generates separate plots for each metric and saves the figure as "history.png".
The filter_out_list method filters out unwanted elements from a given list based on a predefined ignore list.
The predict method performs predictions on test images using the trained model. It loads the model, iterates over each category, selects test images, and predicts their labels. It plots the images with their predicted labels and saves the figures in a specified directory.
The grad_cam_viz method visualizes the class activation heatmap using Grad-CAM. It loads the model, prepares the input image, generates the heatmap using the last convolutional layer, and displays the heatmap along with the original image. The resulting heatmap is saved as an image file.
Overall, the BitVision class provides functionality for training, evaluating, and visualizing a neural network model for bit vision tasks. It encapsulates the necessary preprocessing steps, model assembly, training process, and prediction visualization, making it convenient to use for bit vision applications.
"""


# Package meta-data.
NAME = "drillvision"
DESCRIPTION = long_description
URL = "https://github.com/Atashnezhad/DrillBitVision"
EMAIL = "atashnezhad2@gmail.com"
AUTHOR = "Amin Atashnezhad"
REQUIRES_PYTHON = ">=3.9.0"


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR
PACKAGE_DIR = ROOT_DIR / "neural_network_model"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"regression_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        # "Programming Language :: Python :: Implementation :: CPython",
        # "Programming Language :: Python :: Implementation :: PyPy",
    ],
)

if __name__ == "__main__":
    list_reqs()
