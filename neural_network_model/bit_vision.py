import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from neural_network_model.model import SETTING

# Set seed to get the same random numbers each time
random.seed(1)

# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Ignore all warnings
warnings.filterwarnings("ignore")


class BitVision:
    """
    This class is used to train a neural network model for bit vision.
    """

    def __init__(self, *args, **kwargs):
        self.layer_names: Dict[str, List] = {}
        self.train_test_val_dir: str = kwargs.get(
            "train_test_val_dir",
            SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS,
        )
        self.model: tensorflow.keras.models.Sequential = None
        # by default the following model is assembled
        self.assemble_deep_net_model()
        self.train_val_gens = {}
        self.model_history = None
        self.model_class_indices: Dict[str, int] = {}

    @property
    def categories(self) -> List[str]:
        """
        This function returns the list of categories.
        :return: list of categories
        """
        # get the list of dirs in the resource_dir
        subdir_name = os.listdir(self.train_test_val_dir)
        # check if there is .DS_Store in the subdir_name if so remove it
        subdir_name = self._filter_out_list(list_to_be_edited=subdir_name)
        subdir_path = self.train_test_val_dir / subdir_name[0]
        categories_name = os.listdir(subdir_path)
        # filter the list of categories if there is a case in the IGNORE_LIST
        categories_name = self._filter_out_list(list_to_be_edited=categories_name)
        return categories_name

    @property
    def data_details(self) -> Dict[str, Dict[str, int]]:
        """
        This property check the dirs and make a dict and save the
        train test val data details in the dict.
        """
        resource_dir = (
            self.train_test_val_dir
            or Path(__file__).parent / ".." / self.train_test_val_dir
        )
        # get the list of dirs in the resource_dir
        subdir_name = os.listdir(resource_dir)
        # check if there is .DS_Store in the subdir_name if so remove it
        subdir_name = self._filter_out_list(list_to_be_edited=subdir_name)
        data = {}
        # for each subdir in the resource_dir, get the list of files
        for subdir in subdir_name:
            data[subdir] = {}
            subdir_path = resource_dir / subdir
            subdir_path_lisr = self._filter_out_list(
                list_to_be_edited=os.listdir(subdir_path)
            )
            for category in subdir_path_lisr:
                category_path = subdir_path / category
                #  check if there is .DS_Store in the subdir_name if so remove it
                # category_path = self._filter_out_list(list_to_be_edited=category_path)
                # for each file in the category, find the number of files
                num_files = len(os.listdir(category_path))
                # logger.info(f"Number of files in {category_path.resolve()} is {num_files}")
                data[subdir][category] = num_files
        return data

    def assemble_deep_net_model(self) -> tensorflow.keras.models.Sequential:
        """
        This function is used to assemble the deep net model.
        """
        image_size = SETTING.FLOW_FROM_DIRECTORY_SETTING.IMAGE_SIZE

        model = Sequential()
        activ = "relu"

        model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation=activ,
                input_shape=(image_size, image_size, 3),
            )
        )
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(3, 3), activation=activ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(3, 3), activation=activ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64, activation=activ))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.categories), activation="softmax"))
        self.model = model
        return self.model

    def assemble_deep_net_model_2(self) -> tensorflow.keras.models.Sequential:
        """
        This function is used to assemble the deep net model.
        """

        image_size = SETTING.FLOW_FROM_DIRECTORY_SETTING.IMAGE_SIZE

        kernel_size = (3, 3)
        pool_size = (2, 2)
        first_filters = 32
        second_filters = 64
        third_filters = 128

        dropout_conv = 0.3
        dropout_dense = 0.3

        model = Sequential()
        model.add(
            Conv2D(
                first_filters,
                kernel_size,
                activation="relu",
                input_shape=(image_size, image_size, 3),
            )
        )
        model.add(Conv2D(first_filters, kernel_size, activation="relu"))
        model.add((MaxPooling2D(pool_size=pool_size)))
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(second_filters, kernel_size, activation="relu"))
        model.add(Conv2D(second_filters, kernel_size, activation="relu"))
        model.add((MaxPooling2D(pool_size=pool_size)))
        model.add(Dropout(dropout_conv))

        model.add(Conv2D(third_filters, kernel_size, activation="relu"))
        model.add(Conv2D(third_filters, kernel_size, activation="relu"))
        model.add((MaxPooling2D(pool_size=pool_size)))
        model.add(Dropout(dropout_conv))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(dropout_dense))
        model.add(Dense(len(self.categories), activation="softmax"))

        self.model = model
        return self.model

    def compile_model(self, **kwargs) -> None:
        """
        This function is used to compile the model.
        :return:
        """

        loss = kwargs.get("loss", SETTING.MODEL_SETTING.LOSS)
        metrics = kwargs.get("metrics", SETTING.MODEL_SETTING.METRICS)

        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss,
            metrics=metrics,
        )

        self.model.summary()

        # Initialize the dictionary
        layer_names = {"conv_layer": [], "other_layer": []}

        # Iterate over the model layers and store the names
        for idx in range(len(self.model.layers)):
            layer_name = self.model.get_layer(index=idx).name

            # Check if the layer name contains 'conv' substring
            if "conv" in layer_name.lower():
                layer_names["conv_layer"].append(layer_name)
            else:
                layer_names["other_layer"].append(layer_name)

        self.layer_names = layer_names

    def plot_image_category(self, *args, **kwargs) -> None:
        nrows = kwargs.get("nrows", 1)
        number_of_categories = len(self.categories)
        ncols = kwargs.get("ncols", number_of_categories)
        subdir = kwargs.get("subdir", "train")
        fig_size = kwargs.get("fig_size", (17, 10))

        # Get one image for each category in train data and plot them
        fig, axs = plt.subplots(nrows, ncols, figsize=fig_size)

        if nrows == 1 and ncols > 1:
            axs = axs.reshape(
                (ncols,)
            )  # Reshape axs to handle 1 row and multiple columns

        for i, category in enumerate(self.categories):
            category_path = self.train_test_val_dir / subdir / category
            image_path = category_path / os.listdir(category_path)[1]
            img = load_img(image_path)

            if nrows == 1 and ncols > 1:
                axs[i].imshow(img)  # Use 1D indexing for 1 row and multiple columns
                axs[i].set_title(category)
            else:
                axs[i // ncols, i % ncols].imshow(img)  # Adjusted indexing
                axs[i // ncols, i % ncols].set_title(category)  # Adjusted indexing

        plt.show()

    def _rescaling(self) -> None:
        """
        This function is used to rescale the images.
        The images were augmented before in the preprocessing step.
        Here we just rescale them.
        """
        my_list = ["train", "val"]
        for subdir in my_list:
            datagen = image.ImageDataGenerator(rescale=SETTING.DATA_GEN_SETTING.RESCALE)

            generator = datagen.flow_from_directory(
                directory=self.train_test_val_dir / subdir
                or SETTING.PREPROCESSING_SETTING.TRAIN_TEST_VAL_SPLIT_DIR_ADDRESS
                / subdir,
                target_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE,
                color_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.COLOR_MODE,
                classes=None,
                class_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.CLASS_MODE,
                batch_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.BATCH_SIZE,
                shuffle=True,
                seed=SETTING.RANDOM_SEED_SETTING.SEED,
                save_to_dir=None,
                save_prefix="",
                save_format="png",
                follow_links=False,
                subset=None,
                interpolation="nearest",
            )
            self.train_val_gens[subdir] = generator
            self.model_class_indices = dict(
                zip(generator.class_indices.values(), generator.class_indices.keys())
            )

            logger.info(f"Rescaling {subdir} data, {generator.class_indices}:")

    def _check_points(
        self, model_save_address: str, model_name: str
    ) -> ModelCheckpoint:
        check_point = ModelCheckpoint(
            model_save_address / model_name,
            monitor=SETTING.MODEL_SETTING.MONITOR,
            verbose=SETTING.MODEL_SETTING.CHECK_POINT_VERBOSE,
            save_best_only=SETTING.MODEL_SETTING.SAVE_BEST_ONLY,
            mode=SETTING.MODEL_SETTING.MODE,
            period=SETTING.MODEL_SETTING.PERIOD,
            metrics=SETTING.MODEL_SETTING.METRICS,
        )
        return check_point

    def _calculate_number_from_dict(self, my_dict: dict) -> int:
        total = sum(my_dict.values())
        return total

    def _calculate_class_weight(self, class_counts: dict) -> dict:
        class_labels = list(class_counts.keys())
        class_indices = {label: index for index, label in enumerate(class_labels)}

        total_samples = sum(class_counts.values())
        class_weight = {}
        for class_label, count in class_counts.items():
            class_index = class_indices[class_label]
            class_weight[class_index] = total_samples / (len(class_counts) * count)

        # Normalize the class weights
        class_weight_sum = sum(class_weight.values())
        for class_index in class_weight:
            class_weight[class_index] /= class_weight_sum
        return class_weight

    def train_model(
        self, model_save_address: Path = SETTING.MODEL_SETTING.MODEL_PATH, **kwargs
    ) -> None:
        """
        This function is used to train the model.
        :param model_save_address:
        :param kwargs: model_name, epochs, verbose, class_weight, workers, use_multiprocessing, shuffle
        :return:
        """

        model_name = kwargs.get("model_name", SETTING.MODEL_SETTING.MODEL_NAME)
        epochs = kwargs.get("epochs", SETTING.MODEL_SETTING.EPOCHS)
        verbose = kwargs.get("verbose", SETTING.MODEL_SETTING.FIT_GEN_VERBOSE)
        class_weight = kwargs.get("class_weight", SETTING.MODEL_SETTING.CLASS_WEIGHT)
        workers = kwargs.get("workers", SETTING.MODEL_SETTING.WORKERS)
        use_multiprocessing = kwargs.get(
            "use_multiprocessing", SETTING.MODEL_SETTING.USE_MULTIPROCESSING
        )
        shuffle = kwargs.get("shuffle", SETTING.MODEL_SETTING.SHUFFLE)

        # calculate validation_steps
        BATCH_SIZE = SETTING.FLOW_FROM_DIRECTORY_SETTING.BATCH_SIZE
        TRAINING_SIZE = self._calculate_number_from_dict(self.data_details["train"])
        VALIDATION_SIZE = self._calculate_number_from_dict(self.data_details["val"])

        if not class_weight:
            class_weight = self._calculate_class_weight(self.data_details["train"])

        # We take the ceiling because we do not drop the remainder of the batch
        compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / BATCH_SIZE))
        steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)
        val_steps = compute_steps_per_epoch(VALIDATION_SIZE)

        self._rescaling()
        self.model_history = self.model.fit_generator(
            steps_per_epoch=steps_per_epoch,
            generator=self.train_val_gens["train"],
            epochs=epochs,
            verbose=verbose,
            validation_data=self.train_val_gens["val"],
            validation_steps=val_steps,
            class_weight=class_weight,
            max_queue_size=SETTING.MODEL_SETTING.MAX_QUEUE_SIZE,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=SETTING.MODEL_SETTING.INITIAL_EPOCH,
            callbacks=[self._check_points(model_save_address, model_name)],
        )

        logger.info(f"Model saved to {SETTING.MODEL_SETTING.MODEL_PATH}")

    def plot_history(self, *args, **kwargs):
        """
        This function is used to plot the history of the model.
        :param fig_folder_address: the address of the folder to save the figure
        :return:
        """
        fig_folder_address = kwargs.get(
            "fig_folder_address", SETTING.FIGURE_SETTING.FIG_PRED_OUT_DIR_ADDRESS
        )
        # if the folder does not exist, create it
        if not os.path.exists(fig_folder_address.resolve()):
            os.makedirs(fig_folder_address.resolve())

        logger.info(self.model_history.history.keys())
        keys_plot = ["loss", "accuracy"]
        # make two plots side by side and have train and val for loss and accuracy
        fig, axs = plt.subplots(
            SETTING.FIGURE_SETTING.NUM_ROWS_IN_PLOT_HIST,
            SETTING.FIGURE_SETTING.NUM_COLS_IN_PLOT_HIST,
            figsize=SETTING.FIGURE_SETTING.FIGURE_SIZE_IN_PLOT_HIST,
        )
        for i, key in enumerate(keys_plot):
            axs[i].plot(self.model_history.history[key], color="red")
            axs[i].plot(self.model_history.history[f"val_{key}"], color="green")
            axs[i].set_title(f"model {key}")
            axs[i].set_ylabel(key)
            axs[i].set_xlabel("epoch")
            axs[i].legend(["train", "val"], loc="upper left")
        fig_path = (fig_folder_address / "history.png").resolve()
        plt.savefig(fig_path)
        plt.show()

    @staticmethod
    def _filter_out_list(
        ignore_list: List[str] = SETTING.IGNORE_SETTING.IGNORE_LIST,
        list_to_be_edited: List[str] = None,
    ) -> List[str]:
        for case in ignore_list:
            if case in list_to_be_edited:
                list_to_be_edited.remove(case)
        return list_to_be_edited

    def predict(self, *args, **kwargs):
        """
        This function is used to predict the test data.
        :param args: test_folder_dir needs to have a suborder called test and there needs to have categories folders
        same structure as train_test_val directory
        :param kwargs: fig_save_address: the address of the folder to save the figure,
        model_path: the path of the model to be used for prediction
        :return:
        """
        fig_save_address = kwargs.get(
            "fig_save_address", SETTING.FIGURE_SETTING.FIG_PRED_OUT_DIR_ADDRESS
        )
        # if the folder does not exist, create it
        if not os.path.exists(fig_save_address.resolve()):
            os.makedirs(fig_save_address.resolve())
        model_path = kwargs.get(
            "model_path",
            SETTING.MODEL_SETTING.MODEL_PATH / SETTING.MODEL_SETTING.MODEL_NAME,
        )
        if model_path is None:
            logger.info(f"model_path from SETTING is was used - {model_path}")

        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Default settings
        default_num_rows = SETTING.FIGURE_SETTING.NUM_ROWS_IN_PRED_MODEL
        default_num_cols = SETTING.FIGURE_SETTING.NUM_COLS_IN_PRED_MODEL
        default_figsize = SETTING.FIGURE_SETTING.FIGURE_SIZE_IN_PRED_MODEL

        # Extract details from kwargs or use default settings
        num_rows = kwargs.get("num_rows", default_num_rows)
        num_cols = kwargs.get("num_cols", default_num_cols)
        figure_size = kwargs.get("figure_size", default_figsize)

        for category in self.categories:
            plt.figure(figsize=figure_size)
            number_of_cols = num_cols
            number_of_rows = num_rows
            number_of_test_to_pred = SETTING.MODEL_SETTING.NUMBER_OF_TEST_TO_PRED

            # get the list of test images
            test_images_list = os.listdir(
                self.train_test_val_dir
                / SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES[1]
                / category
            )
            # check if .DS_Store is in the list if so remove it
            test_images_list = self._filter_out_list(list_to_be_edited=test_images_list)

            for i, img in enumerate(test_images_list[0:number_of_test_to_pred]):
                path_to_img = (
                    self.train_test_val_dir
                    / SETTING.PREPROCESSING_SETTING.TRAIN_TEST_SPLIT_DIR_NAMES[1]
                    / category
                    / str(img)
                ).resolve()

                img = load_img(
                    path_to_img,
                    target_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE,
                )
                ax = plt.subplot(number_of_rows, number_of_cols, i + 1)
                plt.imshow(img)
                img = img_to_array(img)
                # expand dimensions to match the shape of model input
                img_batch = np.expand_dims(img, axis=0)
                img_preprocessed = preprocess_input(img_batch)
                # Generate feature output by predicting on the input image
                prediction = model.predict(img_preprocessed)
                prediction = np.argmax(prediction, axis=1)
                logger.info(
                    f"Prediction: {self.model_class_indices.get(prediction[0])}, category: {category}"
                )
                # check if the prediction is correct or not and set the title accordingly
                # and if it is not correct make the color of the title red
                if self.model_class_indices.get(prediction[0]) == category:
                    ax.set_title(
                        f"{self.model_class_indices.get(prediction[0])}", color="green"
                    )
                else:
                    ax.set_title(
                        f"{self.model_class_indices.get(prediction[0])}",
                        color="red",
                    )

            # save the figure in the figures folder
            fig_name = f"prediction_{category}.png"
            fig_path = (fig_save_address / fig_name).resolve()
            if not os.path.exists(fig_path.parent):
                os.makedirs(fig_path.parent)
            plt.savefig(fig_path)
            plt.show()
            plt.tight_layout()

        datagen = image.ImageDataGenerator(SETTING.DATA_GEN_SETTING.RESCALE)
        DoubleCheck_generator = datagen.flow_from_directory(
            directory=self.train_test_val_dir / "test",
            target_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE,
            color_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.COLOR_MODE,
            classes=None,
            class_mode=SETTING.FLOW_FROM_DIRECTORY_SETTING.CLASS_MODE,
            batch_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.BATCH_SIZE,
            shuffle=SETTING.FLOW_FROM_DIRECTORY_SETTING.SHUFFLE,
            seed=SETTING.FLOW_FROM_DIRECTORY_SETTING.SEED,
            save_to_dir=None,
            save_prefix="",
            save_format="png",
            follow_links=False,
            subset=None,
            interpolation="nearest",
        )

        model.evaluate(DoubleCheck_generator)

    # TODO: check the addresses and add as kwargs if needed
    def grad_cam_viz(self, *args, **kwargs):
        model_path = kwargs.get(
            "model_path",
            SETTING.MODEL_SETTING.MODEL_PATH / SETTING.MODEL_SETTING.MODEL_NAME,
        )
        fig_to_save_address = kwargs.get(
            "fig_to_save_address", SETTING.GRAD_CAM_SETTING.IMAGE_NEW_NAME
        )
        gradcam_fig_name = kwargs.get(
            "output_gradcam_fig_name", SETTING.GRAD_CAM_SETTING.GRAD_CAM_FIG_NAME
        )

        img_to_be_applied_path = kwargs.get(
            "img_to_be_applied_path", SETTING.GRAD_CAM_SETTING.IMG_PATH
        )

        print_layer_names = kwargs.get("print_layer_names", False)

        fig_address = fig_to_save_address / gradcam_fig_name
        if model_path is None:
            raise ValueError("model_path is None")

        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        if print_layer_names:
            # print the model layers
            for idx in range(len(model.layers)):
                print(model.get_layer(index=idx).name)

        # model_builder = keras.applications.xception.Xception
        preprocess_input = keras.applications.xception.preprocess_input
        # decode_predictions = keras.applications.xception.decode_predictions

        conv_layer_name_tobe_used = kwargs.get(
            "layer_name", SETTING.GRAD_CAM_SETTING.LAST_CONV_LAYER_NAME
        )

        # The local path to our target image
        img_path = img_to_be_applied_path

        # load the image and show it
        img = load_img(
            img_path, target_size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE
        )
        plt.imshow(img)
        plt.show()

        # Prepare image
        img_array = preprocess_input(
            BitVision._get_img_array(
                img_path, size=SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE
            )
        )
        # Make model
        # model = model_builder(weights="imagenet")
        # Remove last layer's softmax
        model.layers[-1].activation = None
        # Print what the top predicted class is
        # print("Predicted:", decode_predictions(preds, top=1)[0])
        # Generate class activation heatmap
        heatmap = BitVision._make_gradcam_heatmap(
            img_array, model, conv_layer_name_tobe_used
        )

        # Display heatmap
        plt.matshow(heatmap)
        # add the title layer name
        plt.title(conv_layer_name_tobe_used)
        plt.show()
        BitVision._save_and_display_gradcam(img_path, heatmap, cam_path=fig_address)

    @staticmethod
    def _get_img_array(img_path, size):
        # `img` is a PIL image of size 299x299
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array

    @staticmethod
    def _make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    @staticmethod
    def _save_and_display_gradcam(
        img_path,
        heatmap,
        cam_path=SETTING.GRAD_CAM_SETTING.IMAGE_NEW_NAME,
        alpha=SETTING.GRAD_CAM_SETTING.ALPHA,
    ):
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8((1 / SETTING.DATA_GEN_SETTING.RESCALE) * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("hot")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

    @staticmethod
    def return_best_model_name(
        directory: Path = SETTING.MODEL_SETTING.MODEL_PATH,
    ) -> str:
        """
        Return the best model name
        :param directory:
        :return:
        """

        best_model = None
        best_val_acc = 0.0

        for filename in os.listdir(directory):
            if (
                filename.endswith(".h5")
                and "val_acc" in filename
                and "val_accuracy" not in filename
            ):
                # Extract the validation accuracy from the filename
                print(filename.split("_val_acc_")[1].split("_.h5")[0])
                val_acc = float(filename.split("_val_acc_")[1].split("_.h5")[0])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = filename

        return best_model

    @staticmethod
    def print_deepnet_project_metadata():
        SETTING.load_settings_from_json()


if __name__ == "__main__":
    obj = BitVision(
        train_test_val_dir=Path(__file__).parent / ".." / "dataset_train_test_val"
    )
    print(obj.categories)
    print(obj.data_details)
    obj.plot_image_category(nrows=3, ncols=3)
    obj.assemble_deep_net_model_2()
    obj.compile_model()
    obj.train_model(epochs=5)
    obj.plot_history()
    obj.predict(num_rows=2, num_cols=2, figsize=(4, 10))
    print(obj.layer_names)
    for conv_layer in obj.layer_names["conv_layer"]:
        obj.grad_cam_viz(
            gradcam_fig_name="test.png",
            print_layer_names=False,
            test_folder_dir=Path(__file__).parent
            / ".."
            / "dataset_train_test_val"
            / "test",
            layer_name=conv_layer,
        )
