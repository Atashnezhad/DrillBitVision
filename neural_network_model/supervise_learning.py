import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, local_binary_pattern
from skimage.filters import frangi, sobel, threshold_multiotsu
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from tqdm import tqdm

from neural_network_model.model import SUPERVISE_LEARNING_SETTING, TRANSFER_LEARNING_SETTING

# ignore warnings
warnings.filterwarnings("ignore")


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

    @property
    def image_df(self):
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

    def scikit_image_example(self, image_path, **kwargs):
        image = cv2.imread(image_path)
        image = color.rgb2gray(image)#[300:700, 700:900]
        cmap = plt.cm.gray

        plt.rcParams["axes.titlesize"] = "medium"
        axes = plt.figure(figsize=(10, 4)).subplots(2, 9)
        for i, black_ridges in enumerate([True, False]):
            for j, (func, sigmas) in enumerate([
                (self.original, None),
                (meijering, [1]),
                (meijering, range(1, 5)),
                (sato, [1]),
                (sato, range(1, 5)),
                (frangi, [1]),
                (frangi, range(1, 5)),
                (hessian, [1]),
                (hessian, range(1, 5)),
            ]):
                result = func(image, black_ridges=black_ridges, sigmas=sigmas)
                axes[i, j].imshow(result, cmap=cmap)
                if i == 0:
                    title = func.__name__
                    if sigmas:
                        title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
                    axes[i, j].set_title(title)
                if j == 0:
                    axes[i, j].set_ylabel(f'{black_ridges = }')
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        plt.tight_layout()
        plt.show()





    def hessian(self, image, **kwargs):
        h = hessian_matrix(image, **kwargs)
        eigenvals = hessian_matrix_eigvals(h)
        return eigenvals

    def plot_image_subplot(self, subplot_idx, img, title, color=None, cmap="gray"):
        plt.subplot(2, 2, subplot_idx)
        plt.imshow(img, cmap=cmap)
        plt.title(title, color=color)
        plt.axis("off")

    def plot_histogram_subplot(self, subplot_idx, bins, hist, eigenvals, channel_color,
                               plt_log=True):
        plt.subplot(3, 1, subplot_idx)
        plt.bar(
            bins[:-1],
            hist,
            width=(np.max(eigenvals[0]) - np.min(eigenvals[0])) / len(bins[:-1]),
            color=channel_color,
        )
        plt.xlabel("Hessian Value")
        plt.ylabel("Counts")
        plt.title(f"Histogram of Hessian Filter ({channel_color} channel)", color=channel_color.lower())
        if plt_log:
            plt.yscale("log")
        plt.grid(True)

    def hessian_filter_feature_extraction(
        self,
        image_path,
        bins=40,
        cmap="gray",
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        **kwargs,
    ):
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
            channel_titles = ["R", "G", "B"]
            eigenvals = [eigenvals_r, eigenvals_g, eigenvals_b]

            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            for idx, (title, eigenval) in enumerate(zip(channel_titles, eigenvals), start=2):
                self.plot_image_subplot(idx, eigenval[0], f"Hessian Filter ({title} channel)", color=title.lower())

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

            channels = ["R", "G", "B"]
            hist_data = [(hist_r, eigenvals_r, "red"), (hist_g, eigenvals_g, "green"), (hist_b, eigenvals_b, "blue")]

            for idx, (hist, eigenvals, channel_color) in enumerate(hist_data, start=1):
                self.plot_histogram_subplot(idx, bins_r, hist, eigenvals, channel_color,
                                            plt_log=plt_log)

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
        _frangi = frangi(image, **kwargs)
        return _frangi

    def frangi_feature_extraction(
        self, image_path, plt_show=True, plt_log=False, figsize=(10, 10), bins=40,
            cmap="gray", **kwargs
    ):
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
        radius=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.RADIUS,
        n_points=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.NUM_POINTS,
        bins=40,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        width=0.5,
        method=SUPERVISE_LEARNING_SETTING.FILTERS.LOCAl_BINARY_PATTERN.METHOD,
    ):
        image = cv2.imread(image_path)

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Compute LBP for each channel
        lbp_r = local_binary_pattern(r, n_points, radius, method=method)
        lbp_g = local_binary_pattern(g, n_points, radius, method=method)
        lbp_b = local_binary_pattern(b, n_points, radius, method=method)

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
        self,
        image_path,
        bins=40,
        plt_show=False,
        plt_log=False,
        figsize=(10, 10),
        classes=SUPERVISE_LEARNING_SETTING.FILTERS.MULTIOTSU_THRESHOLD.CLASSES,
    ):
        image = cv2.imread(image_path)

        # also apply on whole image
        thresholds = threshold_multiotsu(image, classes=classes)
        # plt the threshold for whole image
        # Applying multi-Otsu threshold for the default value, generating
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(image, bins=thresholds)
        plt.imshow(regions)
        plt.show()

        # Convert the image to RGB if it's in BGR
        if len(image.shape) == 3:  # Check if the image is color (has 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Split the image into R, G, and B channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Apply multi-Otsu thresholding to each channel
        thresholds_r = threshold_multiotsu(r, classes=classes)
        thresholds_g = threshold_multiotsu(g, classes=classes)
        thresholds_b = threshold_multiotsu(b, classes=classes)

        # Convert the thresholded images to binary
        binary_r = r > thresholds_r[1]
        binary_g = g > thresholds_g[1]
        binary_b = b > thresholds_b[1]

        if plt_show:
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
        self, image_path, bins=40, plt_show=False, plt_log=False, figsize=(10, 10), cmap="Greys"
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

        if plt_show:
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
        Filter images based on the eigenvalues of the Hessian matrix
        :param kwargs:  filter_name: str, name of the filter to use
                        dataset_path: str, path to the dataset
                        filtered_dataset_path: str, path to the filtered dataset
        :return:
        """

        cmap = kwargs.get("cmap", "jet")
        filter_name = kwargs.get("filter_name", "hessian")
        dataset_path = kwargs.get("dataset_path", None)
        filtered_dataset_path = kwargs.get("filtered_dataset_path", None)
        replace_existing = kwargs.get("replace_existing", False)  # New parameter

        if filtered_dataset_path is None:
            filtered_dataset_path = str(Path(__file__).parent / ".." / "filtered_dataset")
            Path(filtered_dataset_path).mkdir(parents=True, exist_ok=True)

        if filter_name == "hessian":
            for index, row in tqdm(
                    self.image_df.iterrows(),
                    total=self.image_df.shape[0],
                    desc="Filtering images > hessian"
            ):
                image_path = row["Filepath"]

                image = cv2.imread(image_path)
                eigenvals = self.hessian(image)

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


if __name__ == "__main__":
    obj = SuperviseLearning()

    # print(obj.image_df.head())

    # Load the image
    # image_path = str(
    #     (Path(__file__).parent / ".." / "dataset" / "pdc_bit" / "Image_1.jpg")
    # )

    image_path = str(
        (Path(__file__).parent / ".." / "dataset_ad" / "MildDemented" / "mildDem0.jpg")
    )


    # Apply hessian filter
    # hessian_features = obj.hessian_filter_feature_extraction(
    #     image_path, plt_show=True, plt_log=True,
    # )
    # print(hessian_features)

    # Apply frangi filter
    # frangifeatures = obj.frangi_feature_extraction(
    #     image_path, plt_show=True, plt_log=True
    # )
    # print(frangifeatures)

    # # Apply LBP filter
    # lbp_result = obj.lbp_feature_extraction(image_path, plt_show=True, plt_log=True)
    # print(lbp_result)
    #
    # # # Apply Multi-Otsu thresholding
    # multi_otsu_features = obj.multiotsu_threshold_sk(
    #     image_path, plt_show=True, plt_log=True
    # )
    # print(multi_otsu_features)



    # # Apply Sobel edge detector
    # sobel_features = obj.sobel_edge_detection_sk(
    #     image_path, plt_show=True, plt_log=True, cmap="jet"
    # )
    # print(sobel_features)




    # dataset_path = Path(__file__).parent / ".." / "dataset_ad"
    # obj = SuperviseLearning(dataset_address=dataset_path)
    # obj.filter_images(
    #     dataset_path=dataset_path,
    #     filtered_dataset_path=Path(__file__).parent / ".." / "filtered_dataset_ad",
    #     replace_existing=False,
    #     cmap="seismic",
    # )

    obj.scikit_image_example(image_path)
