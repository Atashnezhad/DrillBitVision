# Implementation based on Kaggle notebook: "Title of Kaggle Notebook"
# Author: Datalira
# Link: https://www.kaggle.com/code/databeru/plant-seedlings-classifier-grad-cam-acc-95
# Location: Dresden, Saxony, Germany

import logging
import os
from pathlib import Path
import matplotlib.cm as cm

import numpy as np
import tensorflow as tf
import math
from IPython.display import Image, display, Markdown
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from neural_network_model.bit_vision import BitVision
from neural_network_model.process_data import Preprocessing


class TransferModel(Preprocessing, BitVision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_df: pd.DataFrame = None

        self._prepare_data()

        self.model: tf.keras.Model = None
        self.model_history: tf.keras.callbacks.History = None

        self.pred: np.ndarray = None

    def _prepare_data(self):
        image_dir = self.dataset_address
        # Get filepaths and labels
        filepaths = list(image_dir.glob(r"**/*.png"))
        # add those with jpg extension
        filepaths.extend(list(image_dir.glob(r"**/*.jpg")))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name="Filepath").astype(str)
        labels = pd.Series(labels, name="Label")

        # Concatenate filepaths and labels
        image_df = pd.concat([filepaths, labels], axis=1)

        # Shuffle the DataFrame and reset index
        image_df = image_df.sample(frac=1).reset_index(drop=True)

        self.image_df = image_df
        # Show the result
        print(self.image_df.head(3))
        # log the data was prepared
        logging.info(f"Data was prepared")

    import math

    def plot_data_images(self, num_rows=None, num_cols=None):

        if not num_rows and not num_cols:
            num_images = len(self.image_df)
            max_plots = 3 * 5  # Maximum number of plots in a 3x5 grid
            num_rows = math.ceil(num_images / 5)
            num_cols = min(num_images, 5)
        else:
            num_images = num_rows * num_cols

        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(15, 10),
            subplot_kw={"xticks": [], "yticks": []},
        )

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                filepath = self.image_df.Filepath[i]
                file_extension = os.path.splitext(filepath)[1].lower()

                if file_extension in [".png", ".jpg"]:
                    try:
                        image = plt.imread(filepath)
                        ax.imshow(image)
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
        plt.show()

    def train_test_split(self, *args, **kwargs):
        # Separate in train and test data
        train_df, test_df = train_test_split(
            self.image_df, train_size=0.9, shuffle=True, random_state=1
        )
        return train_df, test_df

    def _create_gen(self):
        train_df, test_df = self.train_test_split()
        # Load the Images with a generator and Data Augmentation
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            validation_split=0.1,
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )

        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col="Filepath",
            y_col="Label",
            target_size=(224, 224),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
            seed=0,
            subset="training",
            rotation_range=30,  # Uncomment to use data augmentation
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        val_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col="Filepath",
            y_col="Label",
            target_size=(224, 224),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            shuffle=True,
            seed=0,
            subset="validation",
            rotation_range=30,  # Uncomment to use data augmentation
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col="Filepath",
            y_col="Label",
            target_size=(224, 224),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            shuffle=False,
        )

        return train_generator, test_generator, train_images, val_images, test_images

    def _create_model(self):
        # Load the pretrained model
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
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

    def train_model(self, num_categories=2, epochs=10, batch_size=32):
        (
            pretrained_model,
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_model()
        inputs = pretrained_model.input

        x = tf.keras.layers.Dense(128, activation="relu")(pretrained_model.output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        outputs = tf.keras.layers.Dense(num_categories, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        history = model.fit(
            train_images,
            validation_data=val_images,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
        )
        self.model = model
        self.model_history = history

    def plot_metrics_results(self):
        # Create subplots with 1 row and 2 columns
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Plot Accuracy
        axes[0].plot(self.model_history.history["accuracy"])
        axes[0].plot(self.model_history.history["val_accuracy"])
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(["Train", "Validation"])

        # Plot Loss
        axes[1].plot(self.model_history.history["loss"])
        axes[1].plot(self.model_history.history["val_loss"])
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(["Train", "Validation"])

        plt.tight_layout()
        plt.show()

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

    def predcit_test(self):
        (
            train_generator,
            test_generator,
            train_images,
            val_images,
            test_images,
        ) = self._create_gen()

        # Predict the label of the test_images
        pred = self.model.predict(test_images)
        pred = np.argmax(pred, axis=1)

        # Map the label
        labels = train_images.class_indices
        labels = dict((v, k) for k, v in labels.items())
        pred = [labels[k] for k in pred]

        # Display the result
        print(f"The first ... predictions: {pred[:5]}")

        train_df, test_df = self.train_test_split()
        y_test = list(test_df.Label)
        print(classification_report(y_test, pred))

        cf_matrix = confusion_matrix(y_test, pred, normalize="true")
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            cf_matrix,
            annot=True,
            xticklabels=sorted(set(y_test)),
            yticklabels=sorted(set(y_test)),
        )
        plt.title("Normalized Confusion Matrix")
        plt.show()

        self.pred = pred

        # # Display some pictures of the dataset with their labels and the predictions
        # fig, axes = plt.subplots(
        #     nrows=3, ncols=3, figsize=(15, 15), subplot_kw={"xticks": [], "yticks": []}
        # )
        #
        # for i, ax in enumerate(axes.flat):
        #     ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
        #     ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred[i]}")
        # plt.tight_layout()
        # plt.show()

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
        self, img_path, heatmap, cam_path="cam.jpg", alpha=0.4
    ):
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
        superimposed_img.save(cam_path)

        # Display Grad CAM
        #     display(Image(cam_path))

        return cam_path

    def grad_cam_viz(self, *args, **kwargs):
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

        num_rows = kwargs.get("num_rows", None)
        num_cols = kwargs.get("num_cols", None)

        last_conv_layer_name = "Conv_1"
        img_size = (224, 224)

        # Remove last layer's softmax
        self.model.layers[-1].activation = None

        # Display the part of the pictures used by the neural network to classify the pictures
        _, test_df = self.train_test_split()

        if not num_rows and not num_cols:
            # Get the number of rows and columns for subplots
            num_images = len(test_df)
            num_cols = 2
            num_rows = (num_images + num_cols - 1) // num_cols
        else:
            num_images = num_rows * num_cols

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6))

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                img_path = test_df.Filepath.iloc[i]
                img_array = preprocess_input(
                    self._get_img_array(img_path, size=img_size)
                )
                heatmap = self._make_gradcam_heatmap(
                    img_array, self.model, last_conv_layer_name
                )
                cam_path = self._save_and_display_gradcam(img_path, heatmap)
                ax.imshow(plt.imread(cam_path))
                ax.set_title(
                    f"True: {test_df.Label.iloc[i]}\nPredicted: {self.pred[i]}"
                )
            else:
                # Remove unused subplots
                fig.delaxes(ax)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from neural_network_model.process_data import Preprocessing

    # download the dataset
    # obj = Preprocessing()
    # obj.download_images(limit=30)

    transfer_model = TransferModel(
        dataset_address=Path(__file__).parent / ".." / "dataset"
    )
    transfer_model.plot_data_images(num_rows=3, num_cols=3)
    transfer_model.train_model()
    transfer_model.plot_metrics_results()
    transfer_model.results()
    transfer_model.predcit_test()
    transfer_model.grad_cam_viz(num_rows=3, num_cols=2)
