import logging
import os
import random
import shutil
import warnings
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import frangi
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_multiotsu

from skimage.filters import sobel
from skimage.color import rgb2gray


class SuperviseLearning:
    """
    This is a class for Supervise Learning approach
    for image classification.
    """

    def __init__(self, *args, **kwargs):
        """
        The constructor for SuperviseLearning class.

        """
        self.dataset_address = kwargs.get(
            "dataset_address", Path(__file__).parent / ".." / "dataset"
        )

    def hessian_filter(self, image_path):
        image = cv2.imread(image_path)
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Hessian filter
        hessian_result = cv2.Laplacian(image, cv2.CV_64F)

        # Display the result (optional)
        cv2.imshow("Hessian Result", hessian_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sato_filter(self, image_path, bins=40):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply Sato filter
        sato_result = frangi(image)
        # Display the result (optional)
        plt.imshow(sato_result, cmap="gray")
        plt.axis("off")
        plt.show()

    def lbp_filter(self, image_path, radius=3, bins=40, cmap="jet"):
        image = cv2.imread(image_path)
        # Define LBP parameters
        n_points = 8 * radius
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply LBP filter
        lbp_result = local_binary_pattern(image, n_points, radius, method="uniform")

        # Compute the histogram of the LBP result
        hist, bins = np.histogram(lbp_result.ravel(), bins=bins, range=(0, n_points + 2))

        # Display the result (optional)
        plt.imshow(lbp_result, cmap=cmap)
        plt.axis("off")
        plt.show()

        # Display the result (optional)
        plt.subplot(1, 2, 1)
        plt.imshow(lbp_result, cmap=cmap)
        plt.axis("off")

        # Plot the histogram
        plt.subplot(1, 2, 2)
        plt.bar(bins[:-1], hist, width=0.5, align='center')
        plt.xlabel('Pixel Value (Binary)')
        plt.ylabel('Counts')
        plt.title('Histogram of lbp_filter\nThresholded Image')
        plt.tight_layout()
        plt.show()

        # Store the histogram counts per bin as features
        lbp_features = hist.tolist()

        # Now you can use the lbp_features list as the feature representation for the LBP-filtered image.
        return lbp_features


    def multi_otsu_threshold(self, image_path, classes=2, bins=40, cmap="gray"):
        image = cv2.imread(image_path)

        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Multi-Otsu thresholding
        thresholds = threshold_multiotsu(image, classes=classes)
        multi_otsu_output = image >= thresholds

        # Display the result (optional)
        plt.imshow(multi_otsu_output, cmap=cmap)
        plt.axis("off")
        plt.show()

        # extract features, Compute the histogram of the Multi-Otsu thresholded image
        hist, bins = np.histogram(multi_otsu_output, bins=bins)

        # Display the result (optional)
        plt.subplot(1, 2, 1)
        plt.imshow(multi_otsu_output, cmap=cmap)
        plt.axis("off")

        # Plot the histogram
        plt.subplot(1, 2, 2)
        plt.bar(bins[:-1], hist, width=0.5, align='center')
        plt.xlabel('Pixel Value (Binary)')
        plt.ylabel('Counts')
        plt.title('Histogram of Multi-Otsu\nThresholded Image')

        plt.show()

        # Store the histogram counts per bin as features
        multi_otsu_features = hist.tolist()
        return multi_otsu_features

    def sobel_edge_detection_cv2(self, image_path):
        image = cv2.imread(image_path)

        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detector
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Display the original image and Sobel edges side by side (optional)
        cv2.imshow("Original Image", image)
        cv2.imshow("Sobel Edges", gradient_magnitude.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel_edge_detection_sk(self, image_path, bins=40):
        image = cv2.imread(image_path)
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 2:
            image = rgb2gray(image)

        # Apply Sobel edge detector
        sobel_edges = sobel(image)

        # Display the original image and Sobel edges side by side (optional)
        plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        plt.imshow(sobel_edges, cmap='jet')
        plt.title('Sobel Edges')
        plt.axis('off')
        plt.show()

        # Normalize the Sobel edges image to [0, 1]
        sobel_edges = (sobel_edges - np.min(sobel_edges)) / (np.max(sobel_edges) - np.min(sobel_edges))

        # Convert the Sobel edges image to uint8
        sobel_edges_uint8 = (sobel_edges * 255).astype(np.uint8)

        # Compute the histogram of the Sobel edges image
        hist, bins = np.histogram(sobel_edges_uint8, bins=bins, range=(0, 255))

        # Display the histogram (optional)
        plt.bar(bins[:-1], hist, width=5)
        plt.xlabel('Pixel Value')
        plt.ylabel('Counts')
        plt.title('Histogram of Sobel Edges')
        plt.show()

        # Store the histogram counts per bin as features
        _sobel_features = hist.tolist()

        return _sobel_features


if __name__ == "__main__":
    obj = SuperviseLearning()

    # Load the image
    image_path = str((Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_1.png"))

    # Apply Hessian filter
    # obj.hessian_filter(image_path)

    # Apply Sato filter
    # obj.sato_filter(image_path)

    # Apply LBP filter
    lbp_result = obj.lbp_filter(image_path)
    print(len(lbp_result))

    # Apply Multi-Otsu thresholding
    # multi_otsu_features = obj.multi_otsu_threshold(image_path)
    # print(multi_otsu_features)

    # Apply Sobel edge detector
    # sobel_features = obj.sobel_edge_detection_sk(image_path)
    # print(len(sobel_features))
