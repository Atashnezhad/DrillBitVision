import logging
import os
import random
import shutil
import warnings
from pathlib import Path
import cv2
from skimage.filters import frangi
import matplotlib.pyplot as plt


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

    def hessian_filter(self, image):
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

    def sato_filter(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply Sato filter
        sato_result = frangi(image)
        # Display the result (optional)
        plt.imshow(sato_result, cmap="gray")
        plt.axis("off")
        plt.show()

    def filter3(self):
        from skimage.feature import local_binary_pattern

        def lbp_filter(image, radius, n_points):
            # Convert the image to grayscale if it's in color
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply LBP filter
            lbp_output = local_binary_pattern(image, n_points, radius, method="uniform")

            return lbp_output

        # Load the image
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)

        # Define LBP parameters
        radius = 3
        n_points = 8 * radius

        # Apply LBP filter
        lbp_result = lbp_filter(image, radius, n_points)

        # Display the result (optional)
        plt.imshow(lbp_result, cmap="gray")
        plt.axis("off")
        plt.show()

    def filter4(self):
        from skimage.filters import threshold_multiotsu

        def multi_otsu_threshold(image, classes):
            # Convert the image to grayscale if it's in color
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Multi-Otsu thresholding
            thresholds = threshold_multiotsu(image, classes=classes)
            multi_otsu_output = image >= thresholds

            return multi_otsu_output

        # Load the image
        image_path = "path/to/your/image.jpg"
        image = cv2.imread(image_path)

        # Define the number of classes for Multi-Otsu thresholding
        num_classes = 3

        # Apply Multi-Otsu thresholding
        multi_otsu_result = multi_otsu_threshold(image, num_classes)

        # Display the result (optional)
        plt.imshow(multi_otsu_result, cmap="gray")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    obj = SuperviseLearning()

    # Load the image
    image_path = "/Users/amin/Downloads/DrillBitVision/dataset/pdc_bit/Image_1.png"

    # Apply Hessian filter
    # obj.hessian_filter(image_path)

    # Apply Sato filter
    obj.sato_filter(image_path)
