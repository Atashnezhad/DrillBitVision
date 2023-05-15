# Drill Bit Classifier
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-360/)

The Drill Bit Classifier is an app that uses a Convolutional Neural Network (CNN) to 
classify images of drill bits. The app can be used by machinists and engineers to 
quickly and accurately identify the type of drill bit required for a particular job.

# Steps:
1- run the download_iamge.py file to download the images from the web.


Process Module
```mermaid
flowchart LR

A[Download Data\n Bing module] --> B[1-find category names\n 2-make an image dictionary]
B --> C[Augment data] --> D
D[Train Test  Val Split] --> E[Populate images into the\ntrain test val folders] --> F[Train the model]
```


Bit Vision Module
```mermaid
flowchart LR
A[Categories\nproperty] --> B[Data Details\nproperty]
B --> C[Assemble Model] --> D[Compile Model] --> E[Rescale Images\nTrain and Val] 
--> F[Fit Model] --> G[Save Model]
```


[//]: # (# CNN Model Prediction on Test Data)

[//]: # (![alt text]&#40;figures/prediction_pdc_bit.png "Logo Title Text 1"&#41;)

[//]: # (![alt text]&#40;figures/prediction_rollercone_bit.png "Logo Title Text 1"&#41;)


# Grad Cam Heatmap - Rollercone Bit
![alt text](figures/grad_cam_rc_1.png "Logo Title Text 1")

# Grad Cam Heatmap - PDC Bit
![alt text](figures/grad_cam_pdc_1.png "Logo Title Text 1")