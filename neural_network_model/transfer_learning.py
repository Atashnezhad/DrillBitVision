# Implementation based on Kaggle notebook: "Title of Kaggle Notebook"
# Author: Datalira
# Link: https://www.kaggle.com/code/databeru/plant-seedlings-classifier-grad-cam-acc-95
# Location: Dresden, Saxony, Germany

import logging
import math
import os
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imageio.v2 import imread
from IPython.display import Image, Markdown, display
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network_model.bit_vision import BitVision
from neural_network_model.model import TRANSFER_LEARNING_SETTING
from neural_network_model.process_data import Preprocessing

# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TransferModel(Preprocessing, BitVision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_save_path: str = TRANSFER_LEARNING_SETTING.MODEL_SAVE_LOCATION
        self.image_df: pd.DataFrame = None

        self._prepare_data()

        self.model: tf.keras.Model = None
        self.model_history: tf.keras.callbacks.History = None

        self.pred: np.ndarray = None

    def _prepare_data(
        self,
        print_data_head=False,
        x_col=TRANSFER_LEARNING_SETTING.DF_X_COL_NAME,
        y_col=TRANSFER_LEARNING_SETTING.DF_Y_COL_NAME,
    ):
        """
        Prepare the data for the model
        :param print_data_head: print the head of the data
        :param x_col: column name for the filepaths
        :param y_col: column name for the labels
        :return: None
        """
        image_dir = self.dataset_address
        # Get filepaths and labels
        filepaths = list(image_dir.glob(r"**/*.png"))
        # add those with jpg extension
        filepaths.extend(list(image_dir.glob(r"**/*.jpg")))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name=x_col).astype(str)
        labels = pd.Series(labels, name=y_col)

        # Concatenate filepaths and labels
        image_df = pd.concat([filepaths, labels], axis=1)

        # Shuffle the DataFrame and reset index
        image_df = image_df.sample(frac=1).reset_index(drop=True)

        self.image_df = image_df
        if print_data_head:
            print(self.image_df.head(3))
        # log the data was prepared
        logging.info("Data was prepared")

    def plot_classes_number(
        self,
        figsize=(10, 5),
        x_rotation=0,
        palette="Greens_r",
        **kwargs,
    ) -> None:
        """
        Plot the number of images per species
        :param figsize: size of the figure
        :param x_rotation: rotation of the x-axis
        :param palette: color palette
        :param kwargs: figure_folder_path
        :return: None
        """
        figure_folder_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )
        # check if the folder exists if not create it
        if not figure_folder_path.exists():
            os.makedirs(figure_folder_path)

        base_path = self.dataset_address
        subfolders = os.listdir(base_path)
        names = []
        counts = []

        filtered_subfolders = self._filter_out_list(list_to_be_edited=subfolders)

        for folder in filtered_subfolders:
            images = os.listdir(base_path / folder)
            names.append(folder)
            counts.append(len(images))

        counts = np.array(counts)
        names = np.array(names)

        idx = np.argsort(counts)[::-1]

        plt.figure(figsize=figsize)
        sns.barplot(x=names[idx], y=counts[idx], palette=palette)
        plt.xticks(rotation=x_rotation)
        # set y axis label
        plt.ylabel("Number of images")
        plt.title("Number of images per classes")
        plt.tight_layout()
        # save the plot in the figures folder
        plt.savefig(figure_folder_path / "classes_number.png")
        plt.show()

        # print the number of images for each class
        for i in range(len(names)):
            # print(f"number of images belong to {names[i]}: {counts[i]}")
            logger.info(f"number of images belong to {names[i]}: {counts[i]}")

    def analyze_image_names(
        self,
        figsize=(20, 22),
        figsize_2=(10, 7),
        cmap_2="YlGnBu",
        size=15,
        label_size=25,
        num_cluster=5,
        **kwargs,
    ) -> None:
        """
        Analyze the image names if there is any pattern in the names
        :param figsize: size of the figure
        :param figsize_2: size of the second figure
        :param cmap_2: color map of the second figure
        :param size: size of the scatter points
        :param label_size: size of the labels
        :param num_cluster: number of clusters - unsupervised learning on width of images
        """

        figure_folder_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )
        # check if the folder exists if not create it
        if not figure_folder_path.exists():
            os.makedirs(figure_folder_path)

        base_path = self.dataset_address
        subfolders = os.listdir(base_path)
        filtered_subfolders = self._filter_out_list(list_to_be_edited=subfolders)

        total_images = 0
        for folder in filtered_subfolders:
            total_images += len(os.listdir(base_path / folder))

        image_df = pd.DataFrame(
            index=np.arange(0, total_images), columns=["width", "height", "classes"]
        )

        k = 0
        all_images = []
        for m in range(len(subfolders)):
            folder = subfolders[m]

            images = os.listdir(base_path / folder)
            all_images.extend(images)
            n_images = len(images)

            for n in range(0, n_images):
                image = imread(base_path / folder / images[n])
                image_df.loc[k, "width"] = image.shape[0]
                image_df.loc[k, "height"] = image.shape[1]
                image_df.loc[k, "classes"] = folder
                image_df.loc[k, "image_name"] = images[n]
                k += 1

        image_df.width = image_df.width.astype(np.int)
        image_df.height = image_df.height.astype(np.int)
        logger.info(image_df.head())

        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax[0].scatter(image_df.width.values, image_df.height.values, s=size)
        ax[0].set_xlabel("Image width")
        ax[0].set_ylabel("Image height")
        ax[0].set_title("Is image width always equal to image height?")

        for single in image_df.classes.unique():
            # Filter the data based on the 'classes' column
            filtered_data = image_df[image_df.classes == single].width
            plt.hist(filtered_data, density=True, alpha=0.5, label=single)
            # show the legend
            plt.legend()

        ax[1].legend()
        ax[1].set_title("KDE-Plot of image width given classes")
        ax[1].set_xlabel("Image width")
        ax[1].set_ylabel("Density")
        # sns.distplot(image_df.width, ax=ax[2])
        ax[2].set_xlabel("Image width")
        ax[2].set_ylabel("Density")
        ax[2].set_title("Overall image width distribution")

        # set x and y axis font size
        ax[0].tick_params(axis="both", which="major", labelsize=label_size)
        ax[1].tick_params(axis="both", which="major", labelsize=label_size)
        ax[2].tick_params(axis="both", which="major", labelsize=label_size)

        # x and y axis label font size
        for i in range(3):
            ax[i].xaxis.label.set_size(label_size)
            ax[i].yaxis.label.set_size(label_size)
            # tite font size
            ax[i].title.set_size(label_size)

        # show the plot
        plt.tight_layout()
        # save the plot in the figures folder
        plt.savefig(figure_folder_path / "image_width_height.png")
        plt.show()

        # check if there is sort of cluster with respect to the image width
        scaler = StandardScaler()

        X = np.log(image_df[["width", "height"]].values)
        X = scaler.fit_transform(X)

        km = KMeans(
            n_clusters=num_cluster, random_state=TRANSFER_LEARNING_SETTING.RANDOM_STATE
        )
        image_df["cluster_number"] = km.fit_predict(X)

        mean_width_height = (
            image_df.groupby("cluster_number")[["width", "height"]].mean().values
        )
        # Sort the clusters based on the mean width (you can do the same for height if needed)
        cluster_number_order = np.argsort(mean_width_height[:, 0])
        # log for mean_states
        logger.info("Mean states: {}".format(mean_width_height))
        logger.info("Cluster number order: {}".format(cluster_number_order))

        # Create a dictionary to store the information
        cluster_info_dict = {
            "mean_width_height": mean_width_height.tolist(),
            "cluster_number_order": cluster_number_order.tolist(),
        }
        logger.info(f"cluster_info_dict\n {cluster_info_dict}")

        target_leakage = (
            image_df.groupby(["cluster_number", "classes"]).size().unstack().fillna(0)
        )
        target_leakage = target_leakage / image_df.classes.value_counts() * 100
        target_leakage = target_leakage.apply(np.round).astype(np.int)

        plt.figure(figsize=figsize_2)
        sns.heatmap(target_leakage, cmap=cmap_2, annot=True)
        plt.title(
            "The cluster_number is related to the classes!\nThis is based on the images width and height and defined "
            "number of clusters = {}".format(num_cluster)
        )
        plt.tight_layout()
        # save the plot in the figures folder
        plt.savefig(figure_folder_path / "cluster_number_per_class.png")
        plt.show()

    def plot_data_images(
        self, num_rows=None, num_cols=None, figsize=(15, 10), **kwargs
    ):
        """
        Plot the images in a grid
        :param num_rows: number of rows in the grid
        :param num_cols: number of columns in the grid
        :param figsize: size of the figure
        :param kwargs: figure_folder_path
        :return: None
        """
        cmap = kwargs.get("cmap", "gray")
        figure_folder_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )
        # check if the folder exists if not create it
        if not figure_folder_path.exists():
            os.makedirs(figure_folder_path)

        if not num_rows and not num_cols:
            num_images = len(self.image_df)
            # max_plots = 3 * 5  # Maximum number of plots in a 3x5 grid
            num_rows = math.ceil(num_images / 5)
            num_cols = min(num_images, 5)
        else:
            num_images = num_rows * num_cols

        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=figsize,
            subplot_kw={"xticks": [], "yticks": []},
        )

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                filepath = self.image_df.Filepath[i]
                file_extension = os.path.splitext(filepath)[1].lower()

                if file_extension in [".png", ".jpg"]:
                    try:
                        image = plt.imread(filepath)
                        ax.imshow(image, cmap=cmap)
                        ax.set_title(self.image_df.Label[i])
                    except (IOError, SyntaxError):
                        print(f"Error loading image at {filepath}")
                else:
                    print(
                        f"Skipping image at {filepath}. Unsupported file format: {file_extension}"
                    )
            else:
                ax.axis("off")  # Turn off the empty subplot

        plt.tight_layout()
        # save the plot in the figures folder
        plt.savefig(figure_folder_path / "images.png")
        plt.show()

    def _train_test_split(self, *args, **kwargs):
        """
        Split the data into train and test data
        :param args: arguments for the train_test_split function
        train_size: float, int (default is 0.9)
        shuffle: bool (default is True)
        random_state: int (default is 1)
        :return: train_df, test_df
        """
        train_size = kwargs.get("train_size", TRANSFER_LEARNING_SETTING.TRAIN_SIZE)
        shuffle = kwargs.get("shuffle", TRANSFER_LEARNING_SETTING.SHUFFLE)
        random_state = kwargs.get(
            "random_state", TRANSFER_LEARNING_SETTING.RANDOM_STATE
        )
        # Separate in train and test data
        train_df, test_df = train_test_split(
            self.image_df,
            train_size=train_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=self.image_df[
                TRANSFER_LEARNING_SETTING.DF_Y_COL_NAME
            ],  # Add stratify option
        )
        return train_df, test_df

    def _create_gen(self):
        """
        Create the generator for the model
        :return: train_images, val_images, test_images
        """
        train_df, test_df = self._train_test_split()
        # Load the Images with a generator and Data Augmentation
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            validation_split=TRANSFER_LEARNING_SETTING.VALIDATION_SPLIT,
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )

        xcol = TRANSFER_LEARNING_SETTING.DF_X_COL_NAME
        ycol = TRANSFER_LEARNING_SETTING.DF_Y_COL_NAME
        traget_size = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.TARGET_SIZE
        color_mode = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.COLOR_MODE
        class_mode = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.CLASS_MODE
        batch_size = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.BATCH_SIZE
        shuffle = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.SHUFFLE
        seed = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.SEED

        rotation_range = TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.ROTATION_RANGE
        zoom_range = TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.ZOOM_RANGE
        width_shift_range = (
            TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.WIDTH_SHIFT_RANGE
        )
        height_shift_range = (
            TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.HEIGHT_SHIFT_RANGE
        )
        shear_range = TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.SHEAR_RANGE
        horizontal_flip = TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.HORIZONTAL_FLIP
        fill_mode = TRANSFER_LEARNING_SETTING.AUGMENTATION_SETTING.FILL_MODE

        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col=xcol,
            y_col=ycol,
            target_size=traget_size,
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="training",
            rotation_range=rotation_range,  # Uncomment to use data augmentation
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
        )

        val_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col=xcol,
            y_col=ycol,
            target_size=traget_size,
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="validation",
            rotation_range=rotation_range,  # Uncomment to use data augmentation
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col=xcol,
            y_col=ycol,
            target_size=traget_size,
            color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_generator, test_generator, train_images, val_images, test_images

    def _create_model(self):
        """
        The model is a pretrained MobileNetV2 model without the top layer
        :return: pretrained_model, train_generator, test_generator, train_images, val_images, test_images
        """
        x_input = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.IMAGE_SIZE
        y_input = TRANSFER_LEARNING_SETTING.FLOW_FROM_DIRECTORY_SETTING.IMAGE_SIZE

        # Load the pretrained model
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=(x_input, y_input, 3),
            include_top=TRANSFER_LEARNING_SETTING.INCLUDE_TOP,
            weights=TRANSFER_LEARNING_SETTING.WEIGHTS,
            pooling=TRANSFER_LEARNING_SETTING.POOLING,
        )

        pretrained_model.trainable = False

        # Create the generators
        (
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_gen()

        return (
            pretrained_model,
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        )

    def train_model(self, epochs=10, batch_size=32, **kwargs):
        """
        Train the model
        :param epochs: number of epochs
        :param batch_size: batch size
        kwargs:
            model_save_location: location to save the model default is self.model_save_location
            model_name: name of the model default is tf_model.h5
            loss: loss function default is categorical_crossentropy
            metrics:
        :return: None
        """
        if kwargs.get("model_save_path"):
            # check if the path exists if not create it
            if not os.path.exists(kwargs.get("model_save_path")):
                os.makedirs(kwargs.get("model_save_path"))
            self.model_save_path = kwargs.get("model_save_path")
        model_name = kwargs.get("model_name", "tf_model.h5")

        # check number of subfolders in the self.image_df
        folder_path = self.dataset_address  # Replace with the path to your folder
        # Create a Path object for the specified folder
        folder = Path(folder_path)
        # Count the number of subfolders
        num_categories = sum(item.is_dir() for item in folder.iterdir())

        (
            pretrained_model,
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_model()
        inputs = pretrained_model.input

        number_of_units_layer_1 = TRANSFER_LEARNING_SETTING.DENSE_LAYER_1_UNITS
        number_of_units_layer_2 = TRANSFER_LEARNING_SETTING.DENSE_LAYER_2_UNITS
        activation = TRANSFER_LEARNING_SETTING.DENSE_LAYER_ACTIVATION

        x = tf.keras.layers.Dense(number_of_units_layer_1, activation=activation)(
            pretrained_model.output
        )
        x = tf.keras.layers.Dense(number_of_units_layer_2, activation=activation)(x)

        last_layer_activation = TRANSFER_LEARNING_SETTING.LAST_LAYER_ACTIVATION

        outputs = tf.keras.layers.Dense(
            num_categories, activation=last_layer_activation
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = TRANSFER_LEARNING_SETTING.OPTIMIZER

        loss = kwargs.get("loss", TRANSFER_LEARNING_SETTING.LOSS)
        metrics = kwargs.get("metrics", TRANSFER_LEARNING_SETTING.METRICS)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        monitor = TRANSFER_LEARNING_SETTING.MONITOR
        patience = TRANSFER_LEARNING_SETTING.PATIENCE
        restore_best_weights = TRANSFER_LEARNING_SETTING.RESTORE_BEST_WEIGHTS

        history = model.fit(
            train_images,
            validation_data=val_images,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    restore_best_weights=restore_best_weights,
                )
            ],
        )
        self.model = model
        self.model_history = history
        # save the model
        self.model_save_path = os.path.join(self.model_save_path, model_name)
        self.model.save(self.model_save_path)

    def plot_metrics_results(self, **kwargs):
        figure_folder_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )

        figsize = kwargs.get("figsize", (12, 4))
        # check if the folder exists if not create it
        if not figure_folder_path.exists():
            os.makedirs(figure_folder_path)

        # Define the metric names to plot (using the keys present in self.model_history.history)
        metrics_to_plot = {
            key: key.replace("val_", "").capitalize()
            for key in self.model_history.history.keys()
            if key.startswith(
                "val_"
            )  # Only plot if there is a corresponding 'val_' metric
        }

        plt.figure(figsize=figsize)
        for i, (metric, title) in enumerate(metrics_to_plot.items()):
            logger.info(f"Plotting metric: {metric} and title: {title}")

            plt.subplot(1, 2, i + 1)
            train_metric = self.model_history.history[metric.replace("val_", "")]
            val_metric = self.model_history.history[metric]
            plt.plot(
                train_metric, label="Training", color="red", linestyle="--", linewidth=2
            )
            plt.plot(val_metric, label="Validation", color="blue", linewidth=2)
            # if there is string categorical_accuracy in the metric name remove it.
            if "Categorical_accuracy" in title:
                title = title.replace("Categorical_accuracy", "accuracy")

            plt.title("History of {}".format(title), fontweight="bold")
            plt.legend()
            plt.grid()
            plt.ylim(
                0,
            )
            plt.xlabel("Epoch", fontweight="bold")
            plt.ylabel(title, fontweight="bold")
            plt.gca().xaxis.set_major_locator(
                plt.MaxNLocator(integer=True)
            )  # Set x-axis ticks to integers
            plt.tight_layout()

        plt.savefig(figure_folder_path / "metric.png")
        plt.show()
        plt.close()

    @staticmethod
    def printmd(string):
        # Print with Markdowns
        display(Markdown(string))

    def results(self):
        (
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_gen()
        results = self.model.evaluate(test_images, verbose=0)

        TransferModel.printmd(" ## Test Loss: {:.5f}".format(results[0]))
        TransferModel.printmd(
            "## Accuracy on the test set: {:.2f}%".format(results[1] * 100)
        )

        print(" ## Test Loss: {:.5f}".format(results[0]))
        print("## Accuracy on the test set: {:.2f}%".format(results[1] * 100))

    def predict_test(self, model_path: Path = None, **kwargs):
        """
        This method is used to predict the test images using the trained model
        :param model_path: the path of the model to load
        :param kwargs: the arguments to pass to the predict method
        :return: None
        """

        figure_folder_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )
        conf_matx_normalize = kwargs.get("normalize", None)
        cmap = kwargs.get("cmap", None)
        rotation = kwargs.get("rotation", 90)
        y_axis_label_size = kwargs.get("y_axis_label_size", 10)
        x_axis_label_size = kwargs.get("x_axis_label_size", 10)
        title_size = kwargs.get("title_size", 10)
        fig_title = kwargs.get("fig_title", "Confusion Matrix")
        conf_matx_font_size = kwargs.get("conf_matx_font_size", 12)
        custom_titles = kwargs.get("custom_titles", None)

        (
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_gen()

        # Load the model
        # here we check if the model_path path is provided and load it otherwise we use the self.model
        # which is the model trained in the train method in the memory
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)

        # check if the folder exists if not create it
        if not figure_folder_path.exists():
            os.makedirs(figure_folder_path)

        # Predict the label of the test_images
        pred = self.model.predict(test_images)
        pred = np.argmax(pred, axis=1)

        # Map the label
        labels = train_images.class_indices
        labels = dict((v, k) for k, v in labels.items())
        pred = [labels[k] for k in pred]

        # Display the result
        print(f"The first ... predictions: {pred[:5]}\n\n")

        train_df, test_df = self._train_test_split()
        y_test = list(test_df.Label)
        print("classification_report\n", classification_report(y_test, pred))

        cf_matrix = confusion_matrix(y_test, pred, normalize=conf_matx_normalize)
        plt.figure(figsize=(10, 6))

        if custom_titles is not None:
            x_labels = [custom_titles[label] for label in sorted(set(y_test))]
            y_labels = [custom_titles[label] for label in sorted(set(y_test))]
        else:
            x_labels = sorted(set(y_test))
            y_labels = sorted(set(y_test))

        sns.heatmap(
            cf_matrix,
            annot=True,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap=cmap,
            annot_kws={"fontsize": conf_matx_font_size},
        )
        # y axis label rotation 90
        plt.yticks(rotation=rotation)
        # set x and y axis label size
        plt.yticks(fontsize=y_axis_label_size)
        plt.xticks(fontsize=x_axis_label_size)
        # title size
        plt.title(fig_title, fontsize=title_size)
        plt.savefig(figure_folder_path / "confusion_matrix.png")
        plt.show()

        self.pred = pred
        report = classification_report(y_test, pred)
        return report

    def _get_img_array(self, img_path, size):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size "size"
        array = np.expand_dims(array, axis=0)
        return array

    def _make_gradcam_heatmap(
        self, img_array, model, last_conv_layer_name, pred_index=None
    ):
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

    def _save_and_display_gradcam(
        self,
        img_path,
        heatmap,
        cam_name="transf_cam.jpg",
        alpha=0.4,
        **kwargs,
    ):
        """
        Args:
            img_path: path to our image
            heatmap: the heatmap of our image
        Keyword Args:
            cam_name: name of the cam (default: transf_cam.jpg)
            alpha: the alpha of the cam (default: 0.4)
            figure_folder_path: the path to the folder where we want to save the cam (default: figures)
        """

        cam_path = kwargs.get(
            "figure_folder_path", Path(__file__).parent / ".." / "figures"
        )
        # check if the cam_path exists if not create it
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)

        # Load the original image
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path / cam_name)

        # Display Grad CAM
        #     display(Image(cam_path))

        return cam_path

    def grad_cam_viz(self, *args, **kwargs):
        """
        Visualize the Grad-CAM heatmap
        Keyword Arguments:
            num_rows {int} -- Number of rows of the subplot grid (default: {None})
            num_cols {int} -- Number of columns of the subplot grid (default: {None})
            last_conv_layer_name {str} -- Name of the last convolutional layer (default: {"Conv_1"})
            img_size {tuple} -- Size of the image (default: {(224, 224)})
            gard_cam_image_name {str} -- Name of the Grad-CAM image (default: {"transf_cam.jpg"})
            figsize {tuple} -- Size of the figure (default: {(12, 6)})
        """
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

        num_rows = kwargs.get("num_rows", None)
        num_cols = kwargs.get("num_cols", None)
        last_conv_layer_name = kwargs.get("last_conv_layer_name", "Conv_1")
        img_size = kwargs.get("img_size", (224, 224))
        gard_cam_image_name = kwargs.get("gard_cam_image_name", "transf_cam.jpg")
        figsize = kwargs.get("figsize", (8, 6))
        title_lable_size = kwargs.get("title_lable_size", 8)

        # Remove last layer's softmax
        self.model.layers[-1].activation = None

        # Display the part of the pictures used by the neural network to classify the pictures
        _, test_df = self._train_test_split()

        if not num_rows and not num_cols:
            # Get the number of rows and columns for subplots
            num_images = len(test_df)
            num_cols = 2
            num_rows = (num_images + num_cols - 1) // num_cols
        else:
            num_images = num_rows * num_cols

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                # generate a random number between 0 and len(test_df)
                j = np.random.randint(0, len(test_df))
                img_path = test_df.Filepath.iloc[j]
                img_array = preprocess_input(
                    self._get_img_array(img_path, size=img_size)
                )
                heatmap = self._make_gradcam_heatmap(
                    img_array, self.model, last_conv_layer_name
                )
                cam_path = self._save_and_display_gradcam(
                    img_path,
                    heatmap,
                    # cam_name=f"gard_cam_image_name_{i}.jpg"
                )
                ax.imshow(plt.imread(cam_path / f"{gard_cam_image_name}"))
                true_label = test_df.Label.iloc[i]
                predicted_label = self.pred[i]
                title = f"True: {true_label}\nPredicted: {predicted_label}"
                # Set the color of the title based on the prediction
                if true_label == predicted_label:
                    ax.set_title(title, color="green")
                else:
                    ax.set_title(title, color="red")
                # title label size
                ax.title.set_size(title_lable_size)
                # set grid on
                ax.grid(True)

            else:
                # Remove unused subplots
                fig.delaxes(ax)

        plt.tight_layout()
        # save the figure
        plt.savefig(
            Path(__file__).parent / ".." / "figures" / "grad_cam.png",
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    from neural_network_model.process_data import Preprocessing

    # download the dataset
    # obj = Preprocessing()
    # obj.download_images(limit=30)

    transfer_model = TransferModel(
        dataset_address=Path(__file__).parent / ".." / "dataset_ad"
    )

    # transfer_model.plot_classes_number()
    # transfer_model.analyze_image_names()
    transfer_model.plot_data_images(num_rows=3, num_cols=3, cmap="jet")
    # transfer_model.train_model(
    #     epochs=3,
    #     model_save_path=(Path(__file__).parent / ".." / "deep_model").resolve(),
    #     model_name="tf_model_ad_1.h5",
    # )
    # transfer_model.plot_metrics_results()
    # transfer_model.results()
    # one can pass the model address to the predict_test method
    custom_titles = {
        "NonDemented": "Healthy",
        "ModerateDemented": "Moderate",
        "MildDemented": "Mild",
        "VeryMildDemented": "Very Mild",
    }
    transfer_model.predict_test(
        model_path=(
            Path(__file__).parent / ".." / "deep_model" / "tf_model_ad_1.h5"
        ).resolve(),
        rotation=90,
        y_axis_label_size=12,
        x_axis_label_size=12,
        title_size=14,
        fig_title="Original Confusion Matrix",
        conf_matx_font_size=12,
        custom_titles=custom_titles,
        cmap="winter",
        normalize="true",
    )
    transfer_model.grad_cam_viz(num_rows=3, num_cols=2)
