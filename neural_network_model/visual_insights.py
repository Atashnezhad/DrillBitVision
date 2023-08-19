import json
import logging
import multiprocessing
import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, local_binary_pattern
from skimage.filters import frangi, hessian, meijering, sato, sobel, threshold_multiotsu
from sklearn.cluster import KMeans
from tqdm import tqdm

from neural_network_model.model import SUPERVISE_LEARNING_SETTING, TRANSFER_LEARNING_SETTING

# ignore warnings
warnings.filterwarnings("ignore")


class ImageNumeric:
    """
    This is a class for Supervise Learning approach
    for image classification.
    """

    def __init__(self, *args, **kwargs):
        """
        The constructor for Filters class.

        """
        self.dataset_address = kwargs.get(
            "dataset_address", Path(__file__).parent / ".." / "dataset"
        )

    @property
    def image_df(self):
        """
        Generate a pandas DataFrame containing filepaths and labels of images within the specified directory.

        Returns:
            pd.DataFrame: A DataFrame containing filepaths and labels.
        """
        image_dir = self.dataset_address
        # Get filepaths and labels
        filepaths = list(image_dir.glob(r"**/*.png"))
        # add those with jpg extension
        filepaths.extend(list(image_dir.glob(r"**/*.jpg")))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(
            filepaths, name=TRANSFER_LEARNING_SETTING.DF_X_COL_NAME
        ).astype(str)
        labels = pd.Series(labels, name=TRANSFER_LEARNING_SETTING.DF_Y_COL_NAME)

        # Concatenate filepaths and labels
        image_df = pd.concat([filepaths, labels], axis=1)

        # Shuffle the DataFrame and reset index
        image_df = image_df.sample(frac=1).reset_index(drop=True)

        return image_df

    # reference: https://scikit-image.org/docs/stable/auto_examples/edges/
    # plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py
    def original(self, image, **kwargs):
        """Return the original image, ignoring any kwargs."""
        return image

    def is_valid_section(self, section_zoom, image_shape):
        if len(section_zoom) != 4:
            return False

        top, bottom, left, right = section_zoom

        if top >= bottom or left >= right:
            return False

        image_height, image_width = image_shape

        if top < 0 or bottom > image_height or left < 0 or right > image_width:
            return False

        return True

    def scikit_image_example(self, image_path, **kwargs):
        """
        Generate a grid of subplots to showcase the effects of different filters from the scikit-image library
        on a given image.

        Parameters:
            image_path (str): Path to the input image.
            **kwargs: Additional keyword arguments.
            section (list): Section of the image to be processed.

        Returns:
            None
        """
        image = cv2.imread(image_path)
        logging.info(f"image shape: {image.shape}")
        section_zoom: list = kwargs.get("section_zoom", None)

        save_path = kwargs.get("save_path", None)
        save_name = kwargs.get("save_name", None)

        image_shape = image.shape[:2]
        if section_zoom and self.is_valid_section(section_zoom, image_shape):
            # crop the image
            image = image[
                section_zoom[0]: section_zoom[1], section_zoom[2]: section_zoom[3]
            ]
        image = color.rgb2gray(image)
        cmap = plt.cm.gray

        plt.rcParams["axes.titlesize"] = "medium"
        axes = plt.figure(figsize=(10, 4)).subplots(2, 9)
        for i, black_ridges in enumerate([True, False]):
            for j, (func, sigmas) in tqdm(
                enumerate(
                    [
                        (self.original, None),
                        (meijering, [1]),
                        (meijering, range(1, 5)),
                        (sato, [1]),
                        (sato, range(1, 5)),
                        (frangi, [1]),
                        (frangi, range(1, 5)),
                        (hessian, [1]),
                        (hessian, range(1, 5)),
                    ]
                ),
                total=9,
            ):
                result = func(image, black_ridges=black_ridges, sigmas=sigmas)
                axes[i, j].imshow(result, cmap=cmap)
                if i == 0:
                    title = func.__name__
                    if sigmas:
                        title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
                    axes[i, j].set_title(title)
                if j == 0:
                    axes[i, j].set_ylabel(f"{black_ridges = }")
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        # save the figure using save_path and save_name
        if save_path and save_name:
            # check if the save path exists if not create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path / save_name)

        plt.tight_layout()
        plt.show()

    def hessian(self, image, **kwargs):
        """
        Plot an image subplot.

        Parameters:
            subplot_idx (int): Index of the subplot in the overall figure.
            img (numpy.ndarray): Image data to be displayed.
            title (str): Title of the subplot.
            color (str, optional): Color for the title text. Default is None.
            cmap (str, optional): Colormap for displaying the image. Default is "gray".
        """
        h = hessian_matrix(image, **kwargs)
        eigenvals = hessian_matrix_eigvals(h)
        return eigenvals

    def plot_image_subplot(self, subplot_idx, img, title, color=None, cmap="gray"):
        plt.subplot(2, 2, subplot_idx)
        plt.imshow(img, cmap=cmap)
        plt.title(title, color=color)
        plt.axis("off")

    def plot_histogram_subplot(
        self, subplot_idx, bins, hist, eigenvals, channel_color, plt_log=True
    ):
        """
        Plot a histogram subplot for the Hessian filter feature extraction.

        Parameters:
            subplot_idx (int): Index of the subplot in the overall figure.
            bins (numpy.ndarray): Bin edges for histogram computation.
            hist (numpy.ndarray): Histogram values.
            eigenvals (numpy.ndarray): Eigenvalues of the Hessian matrix for the channel.
            channel_color (str): Color of the channel (e.g., "R", "G", "B").
            plt_log (bool, optional): Whether to use a logarithmic y-scale for the histogram. Default is True.
        """
        plt.subplot(3, 1, subplot_idx)
        plt.bar(
            bins[:-1],
            hist,
            width=(np.max(eigenvals[0]) - np.min(eigenvals[0])) / len(bins[:-1]),
            color=channel_color,
        )
        plt.xlabel("Hessian Value")
        plt.ylabel("Counts")
        plt.title(
            f"Histogram of Hessian Filter ({channel_color} channel)",
            color=channel_color.lower(),
        )
        if plt_log:
            plt.yscale("log")
        plt.grid(True)

    def hessian_filter_feature_extraction(
        self,
        image_path,
        bins=SUPERVISE_LEARNING_SETTING.BINS,
        cmap="gray",
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        **kwargs,
    ):
        """
        Extract features using the Hessian filter on an input image.

        The Hessian filter is a powerful tool for detecting structures at various scales in images.

        Parameters:
            image_path (str): Path to the image to be processed.
            bins (int, optional): Number of bins for histogram computation. Default is SUPERVISE_LEARNING_SETTING.BINS.
            cmap (str, optional): Colormap for displaying images. Default is "gray".
            plt_show (bool, optional): Whether to display plots. Default is False.
            plt_log (bool, optional): Whether to use a logarithmic y-scale in histograms. Default is False.
            figsize (tuple, optional): Size of the figure for plotting. Default is (10, 10).
            **kwargs: Additional keyword arguments to be passed to the hessian filter function.

        Returns:
            dict: Dictionary containing the histogram counts for each channel after Hessian filtering.
        """
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Compute Hessian matrix for each channel and eigenvalues
        eigenvals_r = self.hessian(r, **kwargs)
        eigenvals_g = self.hessian(g, **kwargs)
        eigenvals_b = self.hessian(b, **kwargs)

        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Compute eigenvalues of Hessian matrix
        eigenvals = self.hessian(image_gray)

        if plt_show:
            plt.imshow(eigenvals[0], cmap=cmap)
            plt.title("Hessian Filter")
            plt.axis("off")
            plt.show()

        if plt_show:
            plt.suptitle("Hessian", fontsize=20)
            channel_titles = ["R", "G", "B"]
            eigenvals = [eigenvals_r, eigenvals_g, eigenvals_b]

            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            for idx, (title, eigenval) in enumerate(
                zip(channel_titles, eigenvals), start=2
            ):
                self.plot_image_subplot(
                    idx,
                    eigenval[0],
                    f"Hessian Filter ({title} channel)",
                    color=title.lower(),
                )

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

            hist_data = [
                (hist_r, eigenvals_r, "red"),
                (hist_g, eigenvals_g, "green"),
                (hist_b, eigenvals_b, "blue"),
            ]

            for idx, (hist, eigenvals, channel_color) in enumerate(hist_data, start=1):
                self.plot_histogram_subplot(
                    idx, bins_r, hist, eigenvals, channel_color, plt_log=plt_log
                )

            plt.tight_layout()
            plt.show()

        # Store the histogram counts per bin as features for each channel
        _hessian_features = {
            "R_channel": hist_r.tolist(),
            "G_channel": hist_g.tolist(),
            "B_channel": hist_b.tolist(),
        }

        return _hessian_features

    def frangi(self, image, **kwargs):
        """
        Apply the Frangi filter to an input image.

        The Frangi filter enhances vessel-like structures in medical images.

        Parameters:
            image (numpy.ndarray): Input image to which the Frangi filter will be applied.
            **kwargs: Additional keyword arguments to be passed to the frangi filter function.

        Returns:
            numpy.ndarray: Image with Frangi filter applied.
        """
        _frangi = frangi(image, **kwargs)
        return _frangi

    def frangi_feature_extraction(
        self,
        image_path,
        plt_show=True,
        plt_log=False,
        figsize=(10, 10),
        bins=SUPERVISE_LEARNING_SETTING.BINS,
        cmap="gray",
        **kwargs,
    ):
        """
        Apply Frangi filter to an input image and extract histogram features from the filtered results.

        Frangi filter is used to enhance vessel-like structures in medical images.

        Parameters:
            image_path (str): Path to the image to be processed.
            plt_show (bool, optional): Whether to show the plots, by default True.
            plt_log (bool, optional): Whether to use log scale for y-axis in histograms, by default False.
            figsize (tuple, optional): Size of the figure, by default (10, 10).
            bins (int, optional): Number of bins to be used for histograms, by default SUPERVISE_LEARNING_SETTING.BINS.
            cmap (str, optional): Colormap for displaying images, by default "gray".
            **kwargs: Additional keyword arguments to be passed to the frangi filter function.

        Returns:
            dict: Dictionary containing the histogram counts for each channel.
        """
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply Frangi filter to each channel
        frangi_r = self.frangi(r, **kwargs)
        frangi_g = self.frangi(g, **kwargs)
        frangi_b = self.frangi(b, **kwargs)

        frangi_whole = self.frangi(image, **kwargs)

        if plt_show:
            plt.imshow(frangi_whole, cmap=cmap)
            plt.title("Frangi Filter")
            plt.axis("off")
            plt.show()

        if plt_show:
            # fig title
            plt.suptitle("Frangi", fontsize=20)
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

    def local_binary_pattern(
        self,
        image,
        n_points: int,
        radius: int,
        method=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.METHOD,
    ):
        """
        Compute the Local Binary Pattern (LBP) representation of an input grayscale image.

        Local Binary Pattern is a texture operator that labels the pixels of an image by thresholding the neighborhood
        of each pixel and considers the result as a binary number.

        Parameters:
            image (numpy.ndarray): Grayscale image for which LBP is computed.
            n_points (int): Number of points to be used for LBP.
            radius (int): Radius of the circle to be used for LBP.
            method (str, optional): Method to be used for LBP, by default "uniform".

        Returns:
            numpy.ndarray: LBP representation of the input grayscale image.
        """
        lbp = local_binary_pattern(image, n_points, radius, method)
        return lbp

    def lbp_feature_extraction(
        self,
        image_path,
        radius=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.RADIUS,
        n_points=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.NUM_POINTS,
        bins=SUPERVISE_LEARNING_SETTING.BINS,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        width=0.5,
        method=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.METHOD,
        cmap="gray",
    ):
        """
        Perform Local Binary Pattern (LBP) feature extraction on an input image.

        Local Binary Pattern is a texture operator that labels the pixels of an image by thresholding the neighborhood
        of each pixel and considers the result as a binary number.

        Parameters:
            image_path (str): Path to the image to be processed.
            radius (int, optional): Radius of the circle to be used for LBP, by default 3.
            n_points (int, optional): Number of points to be used for LBP, by default 8.
            bins (int, optional): Number of bins to be used for histogram, by default 256.
            plt_show (bool, optional): Whether to show the plots, by default False.
            plt_log (bool, optional): Whether to use a logarithmic scale for the y-axis of histograms, by default False.
            figsize (tuple, optional): Size of the figure for plotting, by default (10, 10).
            width (float, optional): Width of the bars in the histogram, by default 0.5.
            method (str, optional): Method to be used for LBP, by default "uniform".
            cmap (str, optional): Colormap to use for displaying images, by default "gray".

        Returns:
            dict: A dictionary containing the histogram counts for each channel of the LBP-filtered image.
        """
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Compute LBP for each channel
        lbp_r = self.local_binary_pattern(r, n_points, radius, method)
        lbp_g = self.local_binary_pattern(g, n_points, radius, method)
        lbp_b = self.local_binary_pattern(b, n_points, radius, method)

        # make image gray
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = self.local_binary_pattern(image_gray, n_points, radius, method)

        # Compute histograms for each LBP image
        hist_r, bins_r = np.histogram(lbp_r.ravel(), bins=bins, range=(0, n_points + 2))
        hist_g, bins_g = np.histogram(lbp_g.ravel(), bins=bins, range=(0, n_points + 2))
        hist_b, bins_b = np.histogram(lbp_b.ravel(), bins=bins, range=(0, n_points + 2))

        if plt_show:
            plt.imshow(lbp, cmap=cmap)
            plt.title("Local Binary Pattern (LBP) Filter")
            plt.axis("off")
            plt.show()

        if plt_show:
            # fig title
            plt.suptitle("Local Binary Pattern (LBP)", fontsize=20)
            # Display the original image and LBP images side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(lbp_r, cmap=cmap)
            plt.title("LBP (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(lbp_g, cmap=cmap)
            plt.title("LBP (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(lbp_b, cmap=cmap)
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

    def multiotsu_threshold(
        self,
        image,
        classes=SUPERVISE_LEARNING_SETTING.FILTERS.MULTIOTSU_THRESHOLD.CLASSES,
    ):
        """
        Compute multi-Otsu threshold values for an input image.

        This method computes multi-Otsu threshold values for an input image, which can be used to segment the image into
        multiple classes based on the computed thresholds.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: An array containing the computed multi-Otsu threshold values.
        """
        filtered_threshold_multiotsu = threshold_multiotsu(image, classes=classes)
        return filtered_threshold_multiotsu

    def multiotsu_threshold_feature_extraction(
        self,
        image_path,
        bins=SUPERVISE_LEARNING_SETTING.BINS,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        classes=SUPERVISE_LEARNING_SETTING.FILTERS.MULTIOTSU_THRESHOLD.CLASSES,
    ):
        """
        Apply multi-Otsu thresholding to an image and extract histogram features from the thresholded channels.

        This method applies multi-Otsu thresholding to an image and extracts histogram features from the thresholded
        channels (R, G, and B) based on the specified number of classes. It then returns the histogram features.

        Args:
            image_path (str): Path to the input image.
            bins (int): Number of bins for histogram computation.
            plt_show (bool): Whether to display plots (default: False).
            plt_log (bool): Whether to use a logarithmic scale for y-axis in histograms (default: False).
            figsize (tuple): Size of the displayed figure (default: (10, 10)).
            classes (int): Number of classes for multi-Otsu thresholding
            (default: SUPERVISE_LEARNING_SETTING.FILTERS.MULTIOTSU_THRESHOLD.CLASSES).

        Returns:
            dict: A dictionary containing histogram features for each channel (R, G, and B).
        """
        image = cv2.imread(image_path)

        # also apply on whole image
        thresholds = self.multiotsu_threshold(image, classes=classes)
        # plt the threshold for whole image
        # Applying multi-Otsu threshold for the default value, generating
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(image, bins=thresholds)
        # fig title
        plt.suptitle("Multi-Otsu Thresholding", fontsize=20)
        plt.imshow(regions)
        plt.show()

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply multi-Otsu thresholding to each channel
        thresholds_r = self.multiotsu_threshold(r, classes)
        thresholds_g = self.multiotsu_threshold(g, classes)
        thresholds_b = self.multiotsu_threshold(b, classes)

        # Convert the thresholded images to binary
        binary_r = r > thresholds_r[1]
        binary_g = g > thresholds_g[1]
        binary_b = b > thresholds_b[1]

        if plt_show:
            # fig title
            plt.suptitle("Multi-Otsu Thresholding", fontsize=20)
            # Display the original image and binary threshold images side by side (optional)
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

        # Compute the histograms of the binary threshold images
        hist_r, bins_r = np.histogram(binary_r, bins=bins, range=(0, 1))
        hist_g, bins_g = np.histogram(binary_g, bins=bins, range=(0, 1))
        hist_b, bins_b = np.histogram(binary_b, bins=bins, range=(0, 1))

        if plt_show:
            plt.figure(figsize=figsize)
            # fig title
            plt.suptitle("Multi-Otsu Thresholding", fontsize=20)
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

    def sobel_edge(self, image):
        """
        Apply Sobel edge detection filter to a grayscale image.

        This method applies the Sobel edge detection filter to a grayscale image and returns the filtered image.

        Args:
            image (np.ndarray): Grayscale image to which Sobel edge detection will be applied.

        Returns:
            np.ndarray: Filtered image with Sobel edges.
        """
        filtered_sobel = sobel(image)
        return filtered_sobel

    def sobel_edge_detection_sk(
        self,
        image_path,
        bins=SUPERVISE_LEARNING_SETTING.BINS,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        cmap="Greys",
    ):
        """
        Apply Sobel edge detection to an image and compute histograms of the edges in RGB channels.

        This method applies Sobel edge detection to an image and computes histograms of the Sobel edges
        in the red (R), green (G), and blue (B) channels. Histogram features are returned for each channel.

        Args:
            image_path (str): Path to the image for Sobel edge detection.
            bins (int, optional): Number of bins for histogram computation (default is SUPERVISE_LEARNING_SETTING.BINS).
            plt_show (bool, optional): Whether to display plots (default is False).
            plt_log (bool, optional): Whether to use a logarithmic scale for y-axis on histograms (default is False).
            figsize (tuple, optional): Figure size for plots (default is (10, 10)).
            cmap (str, optional): Colormap for visualization (default is 'Greys').

        Returns:
            dict: A dictionary containing histogram features for each channel.
                {
                    "R_channel": [histogram values],
                    "G_channel": [histogram values],
                    "B_channel": [histogram values]
                }
        """
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply Sobel edge detector to each channel
        sobel_edges_r = self.sobel_edge(r)
        sobel_edges_g = self.sobel_edge(g)
        sobel_edges_b = self.sobel_edge(b)

        # make image gray
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobl = self.sobel_edge(image_gray)

        if plt_show:
            plt.imshow(sobl, cmap=cmap)
            plt.title("Sobel Edges")
            plt.axis("off")
            plt.show()

        if plt_show:
            # fig title
            plt.suptitle("Sobel Edge Detection", fontsize=20)
            # Display the original image and Sobel edges side by side (optional)
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(sobel_edges_r, cmap=cmap)
            plt.title("Sobel Edges (R channel)", color="r")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(sobel_edges_g, cmap=cmap)
            plt.title("Sobel Edges (G channel)", color="g")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(sobel_edges_b, cmap=cmap)
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
            # fig title
            plt.suptitle("Histogram of Sobel Edges", fontsize=20)
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

    def filter_images(
        self,
        **kwargs,
    ):
        """
        Apply image filtering based on specified methods and save the filtered images.

        This method filters images based on the chosen filtering methods such as Hessian matrix,
        Frangi filter, or Local Binary Pattern (LBP). Filtered images are saved in a new directory.

        Args:
            **kwargs: Keyword arguments for customization.
                cmap (str, optional): Name of the colormap for visualization (default is 'jet').
                filter_name (str): Name of the filtering method to use ('hessian', 'frangi', or 'lbp').
                dataset_path (str, optional): Path to the directory containing original images.
                filtered_dataset_path (str): Path to the directory where filtered images will be saved.
                replace_existing (bool, optional): Whether to replace existing filtered images (default is False).

        Returns:
        None
        """

        cmap = kwargs.get("cmap", "jet")
        filter_name = kwargs.get("filter_name", "hessian")
        dataset_path = kwargs.get("dataset_path", None)
        filtered_dataset_path = kwargs.get("filtered_dataset_path", None)
        replace_existing = kwargs.get("replace_existing", False)  # New parameter

        if filtered_dataset_path is None:
            filtered_dataset_path = str(
                Path(__file__).parent / ".." / f"{filtered_dataset_path}"
            )
            Path(filtered_dataset_path).mkdir(parents=True, exist_ok=True)

        if filter_name == "hessian":
            for index, row in tqdm(
                self.image_df.iterrows(),
                total=self.image_df.shape[0],
                desc="Filtering images > hessian",
            ):
                image_path = row["Filepath"]

                image = cv2.imread(image_path)
                # Convert the image to RGB if it's in BGR
                if (
                    len(image.shape) == 3
                ):  # Check if the image is color (has 3 channels)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Convert the image to grayscale
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Compute eigenvalues of Hessian matrix
                    eigenvals = self.hessian(image_gray)

                # Get the sub-folder structure from the original image path
                relative_path = Path(image_path).relative_to(dataset_path)
                filtered_image_path = Path(dataset_path) / relative_path
                # Handle replacing existing images
                if not replace_existing:
                    filtered_image_path = Path(filtered_dataset_path) / relative_path
                    # If the filtered image already exists, and we're not replacing, modify the filename
                    filename_parts = filtered_image_path.stem

                    # Determine the original extension
                    orig_extension = Path(image_path).suffix.lower()

                    new_filename = f"{filename_parts}_filtered{orig_extension}"
                    filtered_image_path = filtered_image_path.parent / new_filename

                # Create necessary directories
                filtered_image_path.parent.mkdir(parents=True, exist_ok=True)

                plt.imshow(eigenvals[0], cmap=cmap)
                plt.axis("off")

                # Save the filtered image
                plt.savefig(filtered_image_path, bbox_inches="tight", pad_inches=0)
                plt.close()

        if filter_name == "frangi":
            for index, row in tqdm(
                self.image_df.iterrows(),
                total=self.image_df.shape[0],
                desc="Filtering images > frangi",
            ):
                image_path = row["Filepath"]

                image = cv2.imread(image_path)
                frangi = self.frangi(image)

                # Get the sub-folder structure from the original image path
                relative_path = Path(image_path).relative_to(dataset_path)
                filtered_image_path = Path(dataset_path) / relative_path
                # Handle replacing existing images
                if not replace_existing:
                    filtered_image_path = Path(filtered_dataset_path) / relative_path
                    # If the filtered image already exists, and we're not replacing, modify the filename
                    filename_parts = filtered_image_path.stem

                    # Determine the original extension
                    orig_extension = Path(image_path).suffix.lower()

                    new_filename = f"{filename_parts}_filtered{orig_extension}"
                    filtered_image_path = filtered_image_path.parent / new_filename

                # Create necessary directories
                filtered_image_path.parent.mkdir(parents=True, exist_ok=True)

                plt.imshow(frangi, cmap=cmap)
                plt.axis("off")

                # Save the filtered image
                plt.savefig(filtered_image_path, bbox_inches="tight", pad_inches=0)
                plt.close()

        if filter_name == "lbp":
            for index, row in tqdm(
                self.image_df.iterrows(),
                total=self.image_df.shape[0],
                desc="Filtering images > lbp",
            ):
                image_path = row["Filepath"]

                image = cv2.imread(image_path)
                # make image gray
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                lbp = self.local_binary_pattern(
                    image_gray,
                    n_points=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.NUM_POINTS,
                    radius=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.RADIUS,
                    method=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.METHOD,
                )

                # Get the sub-folder structure from the original image path
                relative_path = Path(image_path).relative_to(dataset_path)
                filtered_image_path = Path(dataset_path) / relative_path
                # Handle replacing existing images
                if not replace_existing:
                    filtered_image_path = Path(filtered_dataset_path) / relative_path
                    # If the filtered image already exists, and we're not replacing, modify the filename
                    filename_parts = filtered_image_path.stem

                    # Determine the original extension
                    orig_extension = Path(image_path).suffix.lower()

                    new_filename = f"{filename_parts}_filtered{orig_extension}"
                    filtered_image_path = filtered_image_path.parent / new_filename

                # Create necessary directories
                filtered_image_path.parent.mkdir(parents=True, exist_ok=True)

                plt.imshow(lbp, cmap=cmap)
                plt.axis("off")

                # Save the filtered image
                plt.savefig(filtered_image_path, bbox_inches="tight", pad_inches=0)
                plt.close()

    def image_segmentation_knn(
        self, image_path, num_clusters=5, plt_show=False, cmap="gray"
    ):
        """
        Apply K-Means clustering for image segmentation and return a colored version of the segmented image.

        This method loads a grayscale image, applies K-Means clustering to segment the image into a specified
        number of clusters, and returns a colored version of the segmented image.

        Args:
            image_path (str): Path to the grayscale image to be segmented.
            num_clusters (int, optional): Number of clusters (colors) for image segmentation (default is 5).
            plt_show (bool, optional): Whether to show plots of the original grayscale and segmented images
            (default is False).

        Returns:
            np.ndarray: Colored version of the segmented image as a NumPy array.
        """

        # Load the grayscale image
        gray_image = Image.open(image_path).convert("L")
        original_image = cv2.imread(image_path)

        # Convert the grayscale image to a numpy array
        gray_array = np.array(gray_image)

        # Reshape the array to a flat 1D array
        flat_array = gray_array.reshape((-1, 1))

        # Number of clusters (colors) to segment the image into
        num_clusters = num_clusters

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(flat_array)
        labels = kmeans.labels_

        # Reshape the labels back to the original image shape
        segmented_labels = labels.reshape(gray_array.shape)

        # Create a colored version of the grayscale image
        colored_image = np.zeros_like(gray_array, dtype=np.uint8)
        for i in range(num_clusters):
            colored_image[segmented_labels == i] = int(255 * (i + 1) / num_clusters)

        if plt_show:
            # Plot the original grayscale image
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")

            # Plot the segmented and colored image
            plt.subplot(1, 2, 2)
            plt.imshow(colored_image, cmap=cmap)  # You can choose a colormap you like
            plt.title("Segmented and Colored Image")

            plt.tight_layout()
            plt.show()

        return colored_image

    def process_image(
        self,
        row,
        cmap,
        dataset_path,
        segmentation_dataset_path,
        replace_existing,
        num_clusters=5,
    ):
        """
        Process and save an image using image segmentation and colormap.

        This method takes an image, applies image segmentation with a specified number of clusters,
        and saves the segmented image with a colormap applied.

        Args:
            row (pandas.Series): Row containing image information from a DataFrame.
            cmap (str): Name of the colormap for visualization.
            dataset_path (str): Path to the directory containing original images.
            segmentation_dataset_path (str): Path to the directory where segmented images will be saved.
            replace_existing (bool): Whether to replace existing segmented images.
            num_clusters (int, optional): Number of clusters for image segmentation (default is 5).

        Returns:
            None
        """

        image_path = row["Filepath"]
        # Get the sub-folder structure from the original image path
        relative_path = Path(image_path).relative_to(dataset_path)
        segmented_image_dir = (
            Path(segmentation_dataset_path) / relative_path.parent
        )  # Create subdirectory
        segmented_image_path = (
            segmented_image_dir / f"{relative_path.stem}_filtered{relative_path.suffix}"
        )
        # Handle replacing existing images
        if not replace_existing:
            segmented_image_path = Path(segmentation_dataset_path) / relative_path
            # If the filtered image already exists, and we're not replacing, modify the filename
            filename_parts = segmented_image_path.stem

            # Determine the original extension
            orig_extension = Path(image_path).suffix.lower()

            new_filename = f"{filename_parts}_filtered{orig_extension}"
            segmented_image_path = segmented_image_path.parent / new_filename

        # Create necessary directories
        segmented_image_path.parent.mkdir(parents=True, exist_ok=True)

        colored_image = self.image_segmentation_knn(
            image_path, num_clusters=num_clusters, plt_show=False
        )

        # Create a separate figure for each process to avoid conflicts
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(colored_image, cmap=cmap)
        plt.xticks([])  # Remove x-axis ticks and labels
        plt.yticks([])  # Remove y-axis ticks and labels

        # Save the filtered image
        plt.savefig(segmented_image_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def image_segmentation(self, num_clusters=5, **kwargs):
        """
        Apply image segmentation to a directory of images using a specified clustering method.

        Args:
            num_clusters (int, optional): Number of clusters for image segmentation (default is 5).
            **kwargs: Keyword arguments for customization.
                cmap (str, optional): Name of the colormap for visualization (default is 'seismic').
                clustering_method (str, optional): Clustering method for image segmentation (default is 'kmean').
                dataset_path (str, optional): Path to the directory containing images.
                segmentation_dataset_path (str): Path to the directory where segmented images will be saved.
                replace_existing (bool, optional): Whether to replace existing segmented images (default is False).

        Returns:
            None
        """

        cmap = kwargs.get("cmap", "seismic")
        img_segmentation = kwargs.get("clustering_method", "kmean")
        dataset_path = kwargs.get("dataset_path", None)
        segmentation_dataset_path = kwargs.get("segmentation_dataset_path")
        replace_existing = kwargs.get("replace_existing", False)  # New parameter

        # if segmentation_dataset_path does not exist, create it
        if not Path(segmentation_dataset_path).exists():
            segmentation_dataset_path = str(
                Path(__file__).parent / ".." / f"{segmentation_dataset_path}"
            )
            Path(segmentation_dataset_path).mkdir(parents=True, exist_ok=True)

        if img_segmentation == "kmean":
            # Use multiprocessing.Pool to parallelize image segmentation
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = []
                for index, row in tqdm(
                    self.image_df.iterrows(),
                    total=self.image_df.shape[0],
                    desc="Segmentation images > kmean",
                ):
                    result = pool.apply_async(
                        self.process_image,
                        args=(
                            row,
                            cmap,
                            dataset_path,
                            segmentation_dataset_path,
                            replace_existing,
                            num_clusters,
                        ),
                    )
                    results.append(result)

                # Wait for all processes to complete
                for result in results:
                    result.wait()

    def apply_colormap_to_directory(self, **kwargs):
        """
        Apply a specified colormap to a directory of images and save the edited images to a new directory.

        Args:
           **kwargs: Keyword arguments for customization.
               cmap (str, optional): Name of the colormap to apply (default is 'seismic').
               dataset_path (str, optional): Path to the original directory containing images.
               edited_dataset_path (str): Path to the new directory where edited images will be saved.
               replace_existing (bool, optional): Whether to replace existing edited images (default is False).

        Returns:
           None
        """

        cmap = kwargs.get("cmap", "seismic")
        dataset_path = kwargs.get("dataset_path", None)
        edited_dataset_path = kwargs.get("edited_dataset_path")
        replace_existing = kwargs.get("replace_existing", False)  # New parameter

        # if edited_dataset_path does not exist, create it
        if not Path(edited_dataset_path).exists():
            edited_dataset_path = str(
                Path(__file__).parent / ".." / f"{edited_dataset_path}"
            )
            Path(edited_dataset_path).mkdir(parents=True, exist_ok=True)

        for index, row in tqdm(
            self.image_df.iterrows(),
            total=self.image_df.shape[0],
            desc="Editing images > cmap",
        ):
            image_path = row["Filepath"]
            # Get the sub-folder structure from the original image path
            relative_path = Path(image_path).relative_to(dataset_path)
            edited_image_path = Path(dataset_path) / relative_path
            # Handle replacing existing images
            if not replace_existing:
                edited_image_path = Path(edited_dataset_path) / relative_path
                # If the filtered image already exists, and we're not replacing, modify the filename
                filename_parts = edited_image_path.stem

                # Determine the original extension
                orig_extension = Path(image_path).suffix.lower()

                new_filename = f"{filename_parts}_filtered{orig_extension}"
                edited_image_path = edited_image_path.parent / new_filename

            # Create necessary directories
            edited_image_path.parent.mkdir(parents=True, exist_ok=True)

            image = plt.imread(image_path)
            plt.imshow(image, cmap=cmap)
            plt.axis("off")
            plt.savefig(edited_image_path, bbox_inches="tight", pad_inches=0)
            plt.close()

    def display_img_class(
        self,
        _random=True,
        selected_imgs=None,
        title_mapping=None,
        arrangement="1x4",
        figsize=(10, 5),
        axes_ticks=True,
        title_show=False,
    ):
        # Get the unique labels
        unique_labels = self.image_df["Label"].unique()

        # Use specified labels from title_mapping if available, otherwise use _random selection
        if title_mapping is not None:
            selected_labels = [
                label for label in title_mapping if label in unique_labels
            ]
        else:
            selected_labels = unique_labels

        if _random:
            self._plot_it(
                selected_labels,
                title_show,
                figsize,
                arrangement,
                title_mapping,
                axes_ticks,
            )
        else:
            # for each img in selected_imgs, find the path from self.image_df
            # and plot it in a figure with the title of the label and arrangement
            # Extract last section of path for each row in the dataframe
            selected_imgs_pathes = {}
            for img in selected_imgs:
                logging.info("Searching for:", img)

                for _path in self.image_df["Filepath"]:
                    edit_path = os.path.basename(_path)
                    if edit_path == img:
                        logging.info("Matched path:", _path)
                        selected_imgs_pathes[img] = _path
                        break
            logging.info("Selected images pathes:", selected_imgs_pathes)
            logging.info("Selected images pathes:", selected_imgs_pathes)

            selected_imgs_pathes = {
                title_mapping[path.split(os.path.sep)[-2]]: path
                for img_name, path in selected_imgs_pathes.items()
            }
            logging.info("Selected images path's:", selected_imgs_pathes)
            # Plot the selected images
            self._plot_it_2(
                arrangement, figsize, title_show, axes_ticks, selected_imgs_pathes
            )

    def _plot_it_2(
        self, arrangement, figsize, title_show, axes_ticks, selected_imgs_pathes
    ):
        rows, cols = map(int, arrangement.split("x"))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if title_show:
            fig.suptitle("Image Display", fontsize=16)

        if rows == 1:
            axes = [axes]

        for i, (label, filepath) in enumerate(selected_imgs_pathes.items()):
            row = i // cols
            col = i % cols
            image = cv2.imread(filepath)
            custom_title = label
            axes[row][col].imshow(image)
            # show the x and y-axis
            if not axes_ticks:
                axes[row][col].axis("off")

            axes[row][col].set_title(custom_title)

        plt.tight_layout()
        plt.show()

    def _plot_it(
        self,
        selected_labels,
        title_show,
        figsize,
        arrangement,
        title_mapping,
        axes_ticks,
    ):
        # Create a dictionary to store selected data
        selected_data = {}
        for label in selected_labels:
            image_df = pd.DataFrame(self.image_df)
            label_data = image_df[image_df["Label"] == label].sample(n=1)
            logging.info(label_data)
            logging.info(label_data["Filepath"].values[0])
            selected_data[label] = label_data["Filepath"].values[0]

        # Convert the dictionary to JSON
        selected_json = json.dumps(selected_data, indent=4)
        logging.info(f"Selected JSON: {selected_json}")
        logging.info(f"Selected JSON: {selected_json}")

        rows, cols = map(int, arrangement.split("x"))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if title_show:
            fig.suptitle("Image Display", fontsize=16)

        if rows == 1:
            axes = [axes]

        for i, (label, filepath) in enumerate(selected_data.items()):
            row = i // cols
            col = i % cols

            image = cv2.imread(filepath)

            # Get the title from the title_mapping dictionary if provided, otherwise use the label itself
            if title_mapping is not None and label in title_mapping:
                custom_title = title_mapping[label]
            else:
                custom_title = label

            axes[row][col].imshow(image)
            # show the x and y-axis
            if not axes_ticks:
                axes[row][col].axis("off")

            axes[row][col].set_title(custom_title)

        plt.tight_layout()
        plt.show()


class RunCodeLocally:
    """
    this is to wrap the lines of code in the name main function
    here I have it in a class so that I can call it from the main.py file
    less clutter in the name main function
    """

    def __init__(self):
        pass

    def run_1(self):
        obj = ImageNumeric()
        print(obj.image_df.head())
        # Load the image
        # image_path = str(
        #     (Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_1.jpg")
        # )
        image_path = str(
            (
                Path(__file__).parent
                / ".."
                / "dataset_ad"
                / "MildDemented"
                / "mildDem0.jpg"
            )
        )

        # Apply hessian filter
        hessian_features = obj.hessian_filter_feature_extraction(
            image_path, plt_show=True, plt_log=True, cmap="seismic"
        )
        print(hessian_features)

        # Apply frangi filter
        frangifeatures = obj.frangi_feature_extraction(
            image_path,
            plt_show=True,
            plt_log=True,
        )
        print(frangifeatures)

        # # Apply LBP filter
        lbp_result = obj.lbp_feature_extraction(image_path, plt_show=True, plt_log=True)
        print(lbp_result)

        # Apply Multi-Otsu thresholding
        multi_otsu_features = obj.multiotsu_threshold_feature_extraction(
            image_path, plt_show=True, plt_log=True
        )
        print(multi_otsu_features)

        # # Apply Sobel edge detector
        sobel_features = obj.sobel_edge_detection_sk(
            image_path, plt_show=True, plt_log=True, cmap="gray"
        )
        print(sobel_features)

    def run_2(self):
        dataset_path = Path(__file__).parent / ".." / "dataset_ad"
        obj = ImageNumeric(dataset_address=dataset_path)

        # followings are code apply to whole directory
        # hessian by default
        obj.filter_images(
            dataset_path=dataset_path,
            filtered_dataset_path=Path(__file__).parent
            / ".."
            / "filtered_dataset_ad_hessian",
            replace_existing=False,
            cmap="seismic",
            filter_name="hessian",
        )
        # obj.filter_images(
        #     dataset_path=dataset_path,
        #     filtered_dataset_path=Path(__file__).parent / ".." / "filtered_dataset_ad_frangi",
        #     replace_existing=False,
        #     cmap="seismic",
        #     filter_name="frangi"
        # )
        # obj.filter_images(
        #     dataset_path=dataset_path,
        #     filtered_dataset_path=Path(__file__).parent / ".." / "filtered_dataset_ad_lbp",
        #     replace_existing=False,
        #     cmap="gray",
        #     filter_name="lbp"
        # )

    def run_3(self):
        dataset_path = Path(__file__).parent / ".." / "dataset"
        obj = ImageNumeric(dataset_address=dataset_path)
        image_path = str(
            (Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_26.jpg")
        )

        obj.scikit_image_example(
            image_path,
            section_zoom=[0, 2000, 0, 1000],
            save_path=Path(__file__).parent / ".." / "assets",
            save_name="scikit_image_example.jpg",
        )

        # only on one image
        # obj.image_segmentation_knn(
        #     image_path, num_clusters=3, plt_show=True, cmap="viridis"
        # )
        # whole directory
        # obj.image_segmentation(
        #     clustering_method="kmean",
        #     dataset_path=dataset_path,
        #     segmentation_dataset_path=Path(__file__).parent / ".." / "segmentation_dataset_ad_kmean_3",
        #     num_clusters=3,
        #     cmap="viridis",
        # )

    def run_4(self):
        obj = ImageNumeric(dataset_address=Path(__file__).parent / ".." / "dataset_ad")

        # Display the images
        # Example title mapping (custom titles for labels)
        custom_titles = {
            "NonDemented": "Healthy",
            "ModerateDemented": "Moderate",
            "MildDemented": "Mild",
            "VeryMildDemented": "Very Mild",
        }
        obj.display_img_class(
            selected_imgs=[
                "nonDem441.jpg",
                "verymildDem1622.jpg",
                "mildDem262.jpg",
                "moderateDem38.jpg",
            ],
            _random=False,
            title_mapping=custom_titles,
            arrangement="1x4",
            figsize=(10, 5),
            title_show=True,
            # axes_ticks=False,
        )

    def run_5(self):
        dataset_path = Path(__file__).parent / ".." / "dataset_ad"
        obj = ImageNumeric(dataset_address=dataset_path)

        obj.apply_colormap_to_directory(
            cmap="seismic",
            dataset_path=dataset_path,
            edited_dataset_path=Path(__file__).parent / ".." / "edited_dataset_ad",
            replace_existing=False,
        )


if __name__ == "__main__":
    run_locally_obj = RunCodeLocally()

    # run_locally_obj.run_1()
    # run_locally_obj.run_2()
    run_locally_obj.run_3()
    # run_locally_obj.run_4()
    # run_locally_obj.run_5()
