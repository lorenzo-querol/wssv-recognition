import tensorflow as tf
import numpy as np
import glob
import cv2
import albumentations as A
import pandas as pd
from rembg import remove, new_session

transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Perspective(p=0.5),
        A.GaussianBlur(p=0.3),
        A.CLAHE(clip_limit=2),
    ]
)


def load_images(path):
    images = []
    for filename in glob.glob(f"{path}/*.jpg"):
        images.append(cv2.imread(filename))
    return images


def augment_images(images, labels, num_aug=7):
    aug_images = []
    aug_labels = []

    for image, label in zip(images, labels):
        aug_images.append(image)
        aug_labels.append(label)

        for i in range(num_aug):
            aug = transforms(image=image)["image"]
            aug_images.append(aug)
            aug_labels.append(label)

    return np.array(aug_images), np.array(aug_labels)


def create_model(base_model, config):
    base_model.trainable = False

    inputs = tf.keras.Input(config["input_shape"])
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(config["dropout_rate"], seed=config["seed_value"])(x)
    outputs = tf.keras.layers.Dense(units=config["num_classes"], activation="softmax")(
        x
    )
    model = tf.keras.Model(inputs, outputs)

    metrics = [
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        tf.keras.metrics.F1Score(name="f1_score", threshold=0.5, average="weighted"),
        tf.keras.metrics.FalseNegatives(name="false_negatives"),
        tf.keras.metrics.TruePositives(name="true_positives"),
        tf.keras.metrics.FalsePositives(name="false_positives"),
        tf.keras.metrics.TrueNegatives(name="true_negatives"),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=metrics,
    )

    return model


def fine_tune(base_model, config: dict, num_layers_to_freeze: int):
    """
    Create a model to fine-tune the last `num_layers_to_freeze` layers of the base model.

    Inputs:
    - base_model: The base model to fine-tune
    - config: The configuration dictionary
    - num_layers_to_freeze: The number of layers to freeze
    """
    base_model.trainable = True

    # Freeze all the layers before the `num_layers_to_freeze` layer
    for layer in base_model.layers[:num_layers_to_freeze]:
        layer.trainable = False

    # Create new model on top.
    inputs = tf.keras.Input(config["input_shape"])
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(config["dropout_rate"], seed=config["seed_value"])(x)
    outputs = (
        tf.keras.layers.Dense(units=config["num_classes"], activation="softmax")(x)
        if config["num_classes"] == 2
        else tf.keras.layers.Dense(units=config["num_classes"])(x)
    )

    model = tf.keras.Model(inputs, outputs)

    # Define metrics
    metrics = [
        tf.keras.metrics.Precision(name="precision", thresholds=0),
        tf.keras.metrics.Recall(name="recall", thresholds=0),
        tf.keras.metrics.F1Score(name="f1_score", threshold=0.5, average="weighted"),
        tf.keras.metrics.FalseNegatives(name="false_negatives", thresholds=0),
        tf.keras.metrics.TruePositives(name="true_positives", thresholds=0),
        tf.keras.metrics.FalsePositives(name="false_positives", thresholds=0),
        tf.keras.metrics.TrueNegatives(name="true_negatives", thresholds=0),
    ]

    loss_fn = (
        tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        if config["num_classes"] == 2
        else tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=loss_fn,
        metrics=metrics,
    )

    return model


def flatten_datasets(dataset):
    """
    Flatten the batches of images and labels into a single array.

    Inputs:
    - dataset: The dataset to flatten. This is a `tf.data.Dataset` object, using the `image_dataset_from_directory` function.
    """
    images, labels = [], []

    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = np.concatenate(images).astype("uint8")
    labels = np.concatenate(labels)

    return images, labels
