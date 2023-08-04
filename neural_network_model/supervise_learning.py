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

    def hessian_filter_feature_extraction(
        self,
        image_path,
        bins=40,
        cmap="jet",
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Compute Hessian matrix for each channel
        H_r = hessian_matrix(r)
        H_g = hessian_matrix(g)
        H_b = hessian_matrix(b)

        # Compute eigenvalues of Hessian matrix for each channel
        eigenvals_r = hessian_matrix_eigvals(H_r)
        eigenvals_g = hessian_matrix_eigvals(H_g)
        eigenvals_b = hessian_matrix_eigvals(H_b)

        if plt_show:
            # Display the original image and Hessian filtered images side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(eigenvals_r[0], cmap=cmap)
            plt.title("Hessian Filter (R channel)", color="red")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(eigenvals_g[0], cmap=cmap)
            plt.title("Hessian Filter (G channel)", color="green")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(eigenvals_b[0], cmap=cmap)
            plt.title("Hessian Filter (B channel)", color="blue")
            plt.axis("off")
            plt.show()

        # Compute histograms for each Hessian filtered image
        hist_r, bins_r = np.histogram(
            eigenvals_r[0].ravel(),
            bins=bins,
            range=(np.min(eigenvals_r[0]), np.max(eigenvals_r[0])),
        )
        hist_g, bins_g = np.histogram(
            eigenvals_g[0].ravel(),
            bins=bins,
            range=(np.min(eigenvals_g[0]), np.max(eigenvals_g[0])),
        )
        hist_b, bins_b = np.histogram(
            eigenvals_b[0].ravel(),
            bins=bins,
            range=(np.min(eigenvals_b[0]), np.max(eigenvals_b[0])),
        )

        if plt_show:
            plt.figure(figsize=figsize)
            # Display the histograms (optional)
            plt.subplot(3, 1, 1)
            plt.bar(
                bins_r[:-1],
                hist_r,
                width=(np.max(eigenvals_r[0]) - np.min(eigenvals_r[0])) / bins,
                color="red",
            )
            plt.xlabel("Hessian Value")
            plt.ylabel("Counts")
            plt.title("Histogram of Hessian Filter (R channel)", color="red")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.bar(
                bins_g[:-1],
                hist_g,
                width=(np.max(eigenvals_g[0]) - np.min(eigenvals_g[0])) / bins,
                color="green",
            )
            plt.xlabel("Hessian Value")
            plt.ylabel("Counts")
            plt.title("Histogram of Hessian Filter (G channel)", color="green")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.bar(
                bins_b[:-1],
                hist_b,
                width=(np.max(eigenvals_b[0]) - np.min(eigenvals_b[0])) / bins,
                color="blue",
            )
            plt.xlabel("Hessian Value")
            plt.ylabel("Counts")
            plt.title("Histogram of Hessian Filter (B channel)", color="blue")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _hessian_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _hessian_features

    def frangi_feature_extraction(
        self, image_path, plt_show=True, plt_log=False, figsize=(10, 10), bins=40
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply Frangi filter to each channel
        frangi_r = frangi(r)
        frangi_g = frangi(g)
        frangi_b = frangi(b)

        if plt_show:
            # Display the original image and Frangi filtered images side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(frangi_r, cmap="gray")
            plt.title("Frangi Filter (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(frangi_g, cmap="gray")
            plt.title("Frangi Filter (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(frangi_b, cmap="gray")
            plt.title("Frangi Filter (B channel)", color="b")
            plt.axis("off")
            plt.show()

        # Compute histograms for each Frangi filtered image
        hist_r, bins_r = np.histogram(
            frangi_r.ravel(), bins=bins, range=(0, np.max(frangi_r))
        )
        hist_g, bins_g = np.histogram(
            frangi_g.ravel(), bins=bins, range=(0, np.max(frangi_g))
        )
        hist_b, bins_b = np.histogram(
            frangi_b.ravel(), bins=bins, range=(0, np.max(frangi_b))
        )

        if plt_show:
            plt.figure(figsize=figsize)
            # Display the histograms (optional)
            plt.subplot(3, 1, 1)
            plt.bar(bins_r[:-1], hist_r, width=np.max(frangi_r) / bins, color="r")
            plt.xlabel("Frangi Value")
            plt.ylabel("Counts")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Frangi Filter (R channel)", color="r")

            plt.subplot(3, 1, 2)
            plt.bar(bins_g[:-1], hist_g, width=np.max(frangi_g) / bins, color="g")
            plt.xlabel("Frangi Value")
            plt.ylabel("Counts")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Frangi Filter (G channel)", color="g")

            plt.subplot(3, 1, 3)
            plt.bar(bins_b[:-1], hist_b, width=np.max(frangi_b) / bins, color="b")
            plt.xlabel("Frangi Value")
            plt.ylabel("Counts")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.title("Histogram of Frangi Filter (B channel)", color="b")

            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _frangi_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _frangi_features

    def lbp_feature_extraction(
        self,
        image_path,
        radius=3,
        n_points=8,
        bins=40,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        width=0.5,
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Compute LBP for each channel
        lbp_r = local_binary_pattern(r, n_points, radius, method="uniform")
        lbp_g = local_binary_pattern(g, n_points, radius, method="uniform")
        lbp_b = local_binary_pattern(b, n_points, radius, method="uniform")

        # Compute histograms for each LBP image
        hist_r, bins_r = np.histogram(lbp_r.ravel(), bins=bins, range=(0, n_points + 2))
        hist_g, bins_g = np.histogram(lbp_g.ravel(), bins=bins, range=(0, n_points + 2))
        hist_b, bins_b = np.histogram(lbp_b.ravel(), bins=bins, range=(0, n_points + 2))

        if plt_show:
            # Display the original image and LBP images side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(lbp_r, cmap="gray")
            plt.title("LBP (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(lbp_g, cmap="gray")
            plt.title("LBP (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(lbp_b, cmap="gray")
            plt.title("LBP (B channel)", color="b")
            plt.axis("off")
            plt.show()

        if plt_show:
            plt.figure(figsize=figsize)
            # Display the histograms (optional)
            plt.subplot(3, 1, 1)
            plt.bar(bins_r[:-1], hist_r, width=width, color="r")
            plt.xlabel("LBP Value")
            plt.ylabel("Counts")
            plt.title("Histogram of LBP (R channel)", color="r")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.subplot(3, 1, 2)
            plt.bar(bins_g[:-1], hist_g, width=width, color="g")
            plt.xlabel("LBP Value")
            plt.ylabel("Counts")
            plt.title("Histogram of LBP (G channel)", color="g")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.bar(bins_b[:-1], hist_b, width=width, color="b")
            plt.xlabel("LBP Value")
            plt.ylabel("Counts")
            plt.title("Histogram of LBP (B channel)", color="b")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _lbp_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _lbp_features

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
            plt.title("Binary Threshold (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(binary_g, cmap="gray")
            plt.title("Binary Threshold (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(binary_b, cmap="gray")
            plt.title("Binary Threshold (B channel)", color="b")
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
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.bar(bins_g[:-1], hist_g, width=0.01, color="green")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of multiotsu threshold_sk (G channel)", color="green")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.bar(bins_b[:-1], hist_b, width=0.01, color="blue")
            plt.xlabel("Pixel Value")
            plt.ylabel("Counts")
            plt.title("Histogram of multiotsu threshold_sk (B channel)", color="blue")
            if plt_log:
                plt.yscale("log")
            plt.grid(True)

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
            plt.title("Sobel Edges (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(sobel_edges_g, cmap="jet")
            plt.title("Sobel Edges (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(sobel_edges_b, cmap="jet")
            plt.title("Sobel Edges (B channel)", color="b")
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

    # Apply hessian filter
    hessian_features = obj.hessian_filter_feature_extraction(
        image_path, plt_show=True, plt_log=True
    )
    print(hessian_features)

    # # # # Apply Sato filter
    # sato_features = obj.frangi_feature_extraction(image_path, plt_show=True, plt_log=True)
    # print(sato_features)

    # # Apply LBP filter
    # lbp_result = obj.lbp_feature_extraction(image_path, plt_show=True)
    # print(lbp_result)

    # # # Apply Multi-Otsu thresholding
    # multi_otsu_features = obj.multiotsu_threshold_sk(
    #     image_path, plt_show=True, plt_log=True
    # )
    # print(multi_otsu_features)

    # # Apply Sobel edge detector
    # sobel_features = obj.sobel_edge_detection_sk(
    #     image_path, plt_show=True, plt_log=True
    # )
    # print(sobel_features)
