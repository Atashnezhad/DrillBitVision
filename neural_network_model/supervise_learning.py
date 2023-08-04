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
        hessian_mat = hessian_matrix(image, sigma=1.0, order="rc")
        eigvals = hessian_matrix_eigvals(hessian_mat)

        # Choose one of the eigenvalues to get the Hessian result
        hessian_output = eigvals[1]  # For instance, using the second eigenvalue

        if plt_show:
            # Display the Hessian result (optional)
            plt.imshow(hessian_output, cmap="gray")
            # set title
            plt.title("Hessian Filter Result")
            plt.axis("off")
            plt.show()

        # Normalize the Hessian-filtered image to [0, 1]
        hessian_output = (hessian_output - np.min(hessian_output)) / (
            np.max(hessian_output) - np.min(hessian_output)
        )

        # Convert the Hessian-filtered image to uint8
        hessian_output_uint8 = (hessian_output * 255).astype(np.uint8)

        # Compute the histogram of the Hessian-filtered image
        hist, bins = np.histogram(hessian_output_uint8, bins=40, range=(0, 255))

        if plt_show:
            # Display the Hessian-filtered image and its histogram side by side
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(hessian_output_uint8, cmap="gray")
            plt.title("Hessian Filtered Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=5)
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of Hessian Filtered Image")
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
        sato_result = (sato_result - np.min(sato_result)) / (
            np.max(sato_result) - np.min(sato_result)
        )

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
            plt.bar(bins[:-1], hist, width=0.5, align="center")
            plt.xlabel("Pixel Value (Binary)")
            plt.ylabel("Counts")
            plt.title("Histogram of sato_filter\nThresholded Image")
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
        hist, bins = np.histogram(
            lbp_result.ravel(), bins=bins, range=(0, n_points + 2)
        )

        if plt_show:
            # Display the result (optional)
            plt.subplot(1, 2, 1)
            # set title
            plt.title("lbp_filter Result")
            plt.imshow(lbp_result, cmap=cmap)
            plt.axis("off")

            # Plot the histogram
            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], hist, width=0.5, align="center")
            plt.xlabel("Pixel Value (Binary)")
            plt.ylabel("Counts")
            plt.title("Histogram of lbp_filter\nThresholded Image")
            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features
        lbp_features = hist.tolist()

        # Now you can use the lbp_features list as the feature representation for the LBP-filtered image.
        return lbp_features

    def multiotsu_threshold_sk(
        self, image_path, bins=40, plt_show=False, plt_log=False, figsize=(10, 10)
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply multi-Otsu thresholding to each channel
        thresholds_r = threshold_multiotsu(r, classes=3)
        thresholds_g = threshold_multiotsu(g, classes=3)
        thresholds_b = threshold_multiotsu(b, classes=3)

        # Convert the thresholded images to binary
        binary_r = r > thresholds_r[1]
        binary_g = g > thresholds_g[1]
        binary_b = b > thresholds_b[1]

        # Combine the binary images from each channel using max (you can use other methods as well)
        binary_combined = np.maximum.reduce([binary_r, binary_g, binary_b])

        if plt_show:
            # Display the original image and binary thresholded images side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.grid(True)
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(binary_r, cmap="gray")
            plt.title("Binary Threshold (R channel)")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(binary_g, cmap="gray")
            plt.title("Binary Threshold (G channel)")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(binary_b, cmap="gray")
            plt.title("Binary Threshold (B channel)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        # Compute the histograms of the binary thresholded images
        hist_r, bins_r = np.histogram(binary_r, bins=bins, range=(0, 1))
        hist_g, bins_g = np.histogram(binary_g, bins=bins, range=(0, 1))
        hist_b, bins_b = np.histogram(binary_b, bins=bins, range=(0, 1))

        if plt_show:
            plt.figure(figsize=figsize)
            # Display the histograms (optional)
            plt.subplot(3, 1, 1)
            plt.bar(bins_r[:-1], hist_r, width=0.01, color="red")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of multiotsu threshold_sk (R channel)", color="red")
            if plt_log:
                plt.yscale("log")

            plt.subplot(3, 1, 2)
            plt.bar(bins_g[:-1], hist_g, width=0.01, color="green")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of multiotsu threshold_sk (G channel)", color="green")
            if plt_log:
                plt.yscale("log")

            plt.subplot(3, 1, 3)
            plt.bar(bins_b[:-1], hist_b, width=0.01, color="blue")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of multiotsu threshold_sk (B channel)", color="blue")
            if plt_log:
                plt.yscale("log")

            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _threshold_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _threshold_features

    def sobel_edge_detection_sk(
        self, image_path, bins=40, plt_show=False, plt_log=False, figsize=(10, 10)
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply Sobel edge detector to each channel
        sobel_edges_r = sobel(r)
        sobel_edges_g = sobel(g)
        sobel_edges_b = sobel(b)

        # Combine the Sobel edges from each channel using max (you can use other methods as well)
        sobel_edges_combined = np.maximum.reduce(
            [sobel_edges_r, sobel_edges_g, sobel_edges_b]
        )

        if plt_show:
            # Display the original image and Sobel edges side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(sobel_edges_r, cmap="jet")
            plt.title("Sobel Edges (R channel)")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(sobel_edges_g, cmap="jet")
            plt.title("Sobel Edges (G channel)")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(sobel_edges_b, cmap="jet")
            plt.title("Sobel Edges (B channel)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        # Normalize the Sobel edges image to [0, 1]
        sobel_edges_r = (sobel_edges_r - np.min(sobel_edges_r)) / (
            np.max(sobel_edges_r) - np.min(sobel_edges_r)
        )
        sobel_edges_g = (sobel_edges_g - np.min(sobel_edges_g)) / (
            np.max(sobel_edges_g) - np.min(sobel_edges_g)
        )
        sobel_edges_b = (sobel_edges_b - np.min(sobel_edges_b)) / (
            np.max(sobel_edges_b) - np.min(sobel_edges_b)
        )

        # Convert the Sobel edges images to uint8
        sobel_edges_r_uint8 = (sobel_edges_r * 255).astype(np.uint8)
        sobel_edges_g_uint8 = (sobel_edges_g * 255).astype(np.uint8)
        sobel_edges_b_uint8 = (sobel_edges_b * 255).astype(np.uint8)

        # Compute the histograms of the Sobel edges images
        hist_r, bins_r = np.histogram(sobel_edges_r_uint8, bins=bins, range=(0, 255))
        hist_g, bins_g = np.histogram(sobel_edges_g_uint8, bins=bins, range=(0, 255))
        hist_b, bins_b = np.histogram(sobel_edges_b_uint8, bins=bins, range=(0, 255))

        if plt_show:
            plt.figure(figsize=figsize)
            # Display the histograms (optional)
            plt.subplot(3, 1, 1)
            plt.bar(bins_r[:-1], hist_r, width=5, color="r")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            if plt_log:
                # y scale is logarithmic
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Sobel Edges (R channel)", color="r")

            plt.subplot(3, 1, 2)
            plt.bar(bins_g[:-1], hist_g, width=5, color="g")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            if plt_log:
                # y scale is logarithmic
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Sobel Edges (G channel)", color="g")

            plt.subplot(3, 1, 3)
            plt.bar(bins_b[:-1], hist_b, width=5, color="b")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            if plt_log:
                # y scale is logarithmic
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Sobel Edges (B channel)", color="b")

            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _sobel_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _sobel_features


if __name__ == "__main__":
    obj = SuperviseLearning()

    # Load the image
    image_path = str(
        (Path(__file__).parent / ".." / "dataset" / "rollercone_bit" / "Image_3.jpg")
    )

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
    multi_otsu_features = obj.multiotsu_threshold_sk(
        image_path, plt_show=True, plt_log=True
    )
    print(multi_otsu_features)

    # # Apply Sobel edge detector
    # sobel_features = obj.sobel_edge_detection_sk(image_path, plt_show=True, plt_log=True)
    # print(sobel_features)
