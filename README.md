# Drill Bit Classifier

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Downloads](https://static.pepy.tech/personalized-badge/drillvision?period=total&units=international_system&left_color=grey&right_color=red&left_text=Downloads)](https://pepy.tech/project/drillvision)
![GitHub CI](https://github.com/Atashnezhad/DrillBitVision/actions/workflows/main.yml/badge.svg)

The Drill Bit Classifier is an app that uses a Convolutional Neural Network (CNN) to 
classify images of drill bits. The app can be used by machinists and engineers to 
quickly and accurately identify the type of drill bit required for a particular job.

## Description:
### Preprocessing Module
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


BitVision is a versatile library initially designed for training, evaluating, and visualizing neural network models specifically tailored to subject with focus on drilling engineering classification tasks. However, it has since evolved to support general classification problems beyond drilling bits. With BitVision, you can effortlessly assemble and train models, make predictions, and generate visualizations for various classification applications, including but not limited to drilling bits.
- **Model Assembly**: BitVision provides methods to assemble deep neural network models for bit vision tasks. You can add convolutional layers, batch normalization, max pooling, dropout, and dense layers based on predefined settings.

- **Data Preparation**: The library handles data preprocessing tasks such as rescaling images using the ImageDataGenerator class from Keras. It also allows you to obtain details about the training, testing, and validation data, including the number of files in each category.

- **Training and Evaluation**: BitVision simplifies the model training process with the fit_generator function. You can specify the number of epochs, validation data, class weights, and utilize ModelCheckpoint to save the best model based on a chosen metric. Additionally, the library provides methods to plot training and validation loss and accuracy over epochs.

- **Prediction Visualization**: With BitVision, you can easily perform predictions on test images using the trained model. The library facilitates plotting images with their predicted labels and saving the figures for analysis and presentation.

- **Grad-CAM Visualization**: BitVision offers functionality to visualize class activation heatmaps using Grad-CAM. You can overlay the heatmaps on the original images and save the resulting visualizations.

### Transfer Learning Module

The code imports necessary libraries and modules, including TensorFlow, NumPy, pandas, seaborn, and matplotlib.
The code defines a class called TransferModel that inherits from two other classes: Preprocessing and BitVision. These classes seem to provide additional functionality for data preprocessing and working with images.
The TransferModel class has several methods for preparing the data, plotting class distributions, analyzing image names, plotting images, performing train-test split, creating data generators, and creating the model.
The TransferModel class uses the MobileNetV2 architecture for transfer learning. It includes methods for creating data generators using ImageDataGenerator from TensorFlow and training the model.


[//]: # (### Process Module)

[//]: # (```mermaid)

[//]: # (flowchart LR)

[//]: # ()
[//]: # (A[Download Data\n Bing module] --> B[1-find category names\n 2-make an image dictionary])

[//]: # (B --> C[Augment data] --> D)

[//]: # (D[Train Test  Val Split] --> E[Populate images into the\ntrain test val folders] --> F[Train the model])

[//]: # (```)


[//]: # (### Bit Vision Module)

[//]: # (```mermaid)

[//]: # (flowchart LR)

[//]: # (A[Categories\nproperty] --> B[Data Details\nproperty])

[//]: # (B --> C[Assemble Model] --> D[Compile Model] --> E[Rescale Images\nTrain and Val] )

[//]: # (--> F[Fit Model] --> G[Save Model])

[//]: # (```)

## Grad Cam Heatmap - Rollercone Bit
![alt text](figures/grad_cam_rc_1.png "Logo Title Text 1")

## Grad Cam Heatmap - PDC Bit
![alt text](figures/grad_cam_pdc_1.png "Logo Title Text 1")


# How to use the Drill Bit Classifier Example
## Installation
```bash
pip install drillvision
```
## Usage
```python
from pathlib import Path
from neural_network_model.process_data import Preprocessing
from neural_network_model.bit_vision import BitVision


if __name__ == "__main__":
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
        img_to_be_applied_path=Path(__file__).parent / "dataset_train_test_val" / "test" / "pdc_bit" / list_of_images[0],
        output_gradcam_fig_name="gradcam.png"
    )
```

## Using TransferLearning Module
```python
from neural_network_model.transfer_learning import TransferModel
from pathlib import Path


transfer_model = TransferModel(
        dataset_address=Path(__file__).parent / "dataset"
    )

transfer_model.plot_classes_number()
transfer_model.analyze_image_names()
transfer_model.plot_data_images(num_rows=3, num_cols=3)
transfer_model.train_model()
transfer_model.plot_metrics_results()
transfer_model.results()
transfer_model.predcit_test()
transfer_model.grad_cam_viz(num_rows=3, num_cols=2)
```

Note that the dataset structure should be as follows:
```
├── dataset
│   ├── class 1
│   └── class 2
│   └── class 3
│   └── class .
│   └── class .
│   └── class .
│   └── class N      
```
