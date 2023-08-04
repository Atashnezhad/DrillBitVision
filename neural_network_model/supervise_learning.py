import logging
import os
import random
import shutil
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import frangi
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_multiotsu

from skimage.filters import sobel
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
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

    def hessian_filter_skimage(self, image_path, plt_show=False):

        image = cv2.imread(image_path)
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = rgb2gray(image)

        # Calculate the Hessian matrix and its eigenvalues
        hessian_mat = hessian_matrix(image, sigma=1.0, order='rc')
        eigvals = hessian_matrix_eigvals(hessian_mat)

        # Choose one of the eigenvalues to get the Hessian result
        hessian_output = eigvals[1]  # For instance, using the second eigenvalue

        if plt_show:
            # Display the Hessian result (optional)
            plt.imshow(hessian_output, cmap='gray')
            # set title
            plt.title("Hessian Filter Result")
            plt.axis('off')
            plt.show()

        # Normalize the Hessian-filtered image to [0, 1]
        hessian_output = (hessian_output - np.min(hessian_output)) / (np.max(hessian_output) - np.min(hessian_output))

        # Convert the Hessian-filtered image to uint8
        hessian_output_uint8 = (hessian_output * 255).astype(np.uint8)

        # Compute the histogram of the Hessian-filtered image
        hist, bins = np.histogram(hessian_output_uint8, bins=40, range=(0, 255))

        if plt_show:
            # Display the Hessian-filtered image and its histogram side by side
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(hessian_output_uint8, cmap='gray')
            plt.title('Hessian Filtered Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=5)
            plt.xlabel('Pixel Value')
            plt.ylabel('Counts')
            plt.title('Histogram of Hessian Filtered Image')
            plt.show()

        Hessian_features = hist.tolist()

        # Return the histogram counts as features
        return Hessian_features

    def sato_filter(self, image_path, bins=40, cmap="jet", plt_show=False):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Sato filter
        sato_result = frangi(image)

        if plt_show:
            # Display the result (optional)
            plt.imshow(sato_result, cmap=cmap)
            # set title
            plt.title("Sato Filter Result")
            plt.axis("off")
            plt.show()

        # Normalize the Sato-filtered image to [0, 1]
        sato_result = (sato_result - np.min(sato_result)) / (np.max(sato_result) - np.min(sato_result))

        # Convert the Sato-filtered image to uint8
        sato_result_uint8 = (sato_result * 255).astype(np.uint8)

        # Compute the histogram of the Sato-filtered image
        hist, bins = np.histogram(sato_result_uint8, bins=bins, range=(0, 255))

        if plt_show:
            # Display the result (optional)
            plt.subplot(1, 2, 1)
            plt.imshow(sato_result, cmap=cmap)
            plt.title("Sato Filter Result")
            plt.axis("off")

            # Plot the histogram
            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=0.5, align='center')
            plt.xlabel('Pixel Value (Binary)')
            plt.ylabel('Counts')
            plt.title('Histogram of sato_filter\nThresholded Image')
            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features
        sato_features = hist.tolist()

        # Now you can use the sato_features list as the feature representation for the Sato-filtered image.
        return sato_features

    def lbp_filter(self, image_path, radius=3, bins=40, cmap="jet", plt_show=False):
        image = cv2.imread(image_path)
        # Define LBP parameters
        n_points = 8 * radius
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply LBP filter
        lbp_result = local_binary_pattern(image, n_points, radius, method="uniform")

        if plt_show:
            # Display the result (optional)
            plt.imshow(lbp_result, cmap=cmap)
            # set title
            plt.title("lbp_filter Result")
            plt.axis("off")
            plt.show()

        # Compute the histogram of the LBP result
        hist, bins = np.histogram(lbp_result.ravel(), bins=bins, range=(0, n_points + 2))

        if plt_show:
            # Display the result (optional)
            plt.subplot(1, 2, 1)
            # set title
            plt.title("lbp_filter Result")
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

    def multi_otsu_threshold(self, image_path, classes=2, bins=40, cmap="gray", plt_show=False):
        image = cv2.imread(image_path)

        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Multi-Otsu thresholding
        thresholds = threshold_multiotsu(image, classes=classes)
        multi_otsu_output = image >= thresholds

        if plt_show:
            # Display the result (optional)
            plt.imshow(multi_otsu_output, cmap=cmap)
            # set title
            plt.title("multi_otsu_threshold Result")
            plt.axis("off")
            plt.show()

        # extract features, Compute the histogram of the Multi-Otsu thresholded image
        hist, bins = np.histogram(multi_otsu_output, bins=bins)

        if plt_show:
            # Display the result (optional)
            plt.subplot(1, 2, 1)
            plt.imshow(multi_otsu_output, cmap=cmap)
            plt.title("multi_otsu_threshold Result")
            plt.axis("off")

            # Plot the histogram
            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=0.5, align='center')
            plt.xlabel('Pixel Value (Binary)')
            plt.ylabel('Counts')
            plt.title('Histogram of Multi-Otsu\nThresholded Image')
            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features
        multi_otsu_features = hist.tolist()
        return multi_otsu_features

    def sobel_edge_detection_sk(self, image_path, bins=40, plt_show=False):
        image = cv2.imread(image_path)
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 2:
            image = rgb2gray(image)

        # Apply Sobel edge detector
        sobel_edges = sobel(image)

        if plt_show:
            # Display the original image and Sobel edges side by side (optional)
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

        if plt_show:
            plt.subplot(1, 2, 1)
            plt.imshow(sobel_edges, cmap='jet')
            plt.title('Sobel Edges')
            plt.axis('off')

            # Display the histogram (optional)
            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=5)
            plt.xlabel('Pixel Value')
            plt.ylabel('Counts')
            plt.title('Histogram of Sobel Edges')
            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features
        _sobel_features = hist.tolist()

        return _sobel_features


if __name__ == "__main__":
    obj = SuperviseLearning()

    # Load the image
    image_path = str((Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_1.png"))

    # # Apply hessian filter
    # hessian_features = obj.hessian_filter_skimage(image_path, plt_show=False)
    # print(hessian_features)

    # # # Apply Sato filter
    # sato_features = obj.sato_filter(image_path, plt_show=False)
    # print(sato_features)
    #
    # # Apply LBP filter
    # lbp_result = obj.lbp_filter(image_path)
    # print(lbp_result)
    #
    # # Apply Multi-Otsu thresholding
    # multi_otsu_features = obj.multi_otsu_threshold(image_path)
    # print(multi_otsu_features)

    # Apply Sobel edge detector
    sobel_features = obj.sobel_edge_detection_sk(image_path)
    print(sobel_features)
