import os
import shutil
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from utils import augment_images, flatten_datasets, fine_tune
from config import models


def train(flag: int, data_dir: str, config: dict):
    """
    Train the model using K-fold cross-validation. The model weights are saved in the checkpoints folder.
    The training logs are saved in the results folder. The model with the lowest validation loss is saved.

    Inputs:
    - flag: An integer to select the base model and model name
    - data_dir: Path to the dataset directory
    """

    train_set, test_set = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        seed=config["seed_value"],
        image_size=config["img_shape"],
        batch_size=config["batch_size"],
        label_mode="binary",
        subset="both",
    )

    train_images, train_labels = flatten_datasets(train_set)

    base_model = models[flag]["base_model"]
    model_name = models[flag]["model_name"]
    num_layers_to_freeze = models[flag]["num_layers_to_freeze"]
    print(
        f"Training {model_name} for {config['epochs']} epochs with {config['batch_size']} batch size, dropout rate of {config['dropout_rate']}, and learning rate of {config['learning_rate']}..."
    )
    print(f"Freezing {num_layers_to_freeze} layers...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed_value"])

    # Results folder
    path = f"results/{model_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    # Loop over the dataset to create separate folds
    for i, (train_idx, valid_idx) in enumerate(
        cv.split(train_images, np.argmax(train_labels, axis=1))
    ):
        print(f"\nFold {i + 1}")

        # Create a new model instance
        model = fine_tune(base_model, config, num_layers_to_freeze)

        # Get the training and validation data
        print("Splitting data...")
        X_train, X_valid = train_images[train_idx, :], train_images[valid_idx, :]
        y_train, y_valid = train_labels[train_idx], train_labels[valid_idx]

        # Augment ONLY training data
        print("Augmenting images...")
        X_train, y_train = augment_images(X_train, y_train, 5)

        # Convert your numpy arrays to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

        # Shuffle, batch, and prefetch the training data
        train_dataset = train_dataset.batch(config["batch_size"]).prefetch(
            tf.data.AUTOTUNE
        )

        # Batch and prefetch the validation data
        valid_dataset = valid_dataset.batch(config["batch_size"]).prefetch(
            tf.data.AUTOTUNE
        )

        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        total = pos + neg
        weights = {0: (1 / neg) * (total) / 2.0, 1: (1 / pos) * (total) / 2.0}

        # Checkpoints folder
        path = f"checkpoints/{model_name}/fold_{i+1}"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config["es_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.CSVLogger(
                f"results/{model_name}/fold_{i+1}.csv", separator=",", append=False
            ),
        ]

        # Fit the model on the train set and evaluate on the validation set
        model.fit(
            train_dataset,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            class_weight=weights,
            validation_data=valid_dataset,
            verbose=1,
            callbacks=callbacks,
        )

        # Save the model weights
        path = f"checkpoints/{model_name}/fold_{i+1}"
        model.save_weights(path + f"/best_weights.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flag",
        type=int,
        default=1,
        help="An integer to select the model. (1) MobilenetV3small, (2) EfficientNetV2B0",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the dataset directory",
    )

    args = parser.parse_args()
    flag = args.flag
    data_dir = args.data_dir

    train(flag, data_dir, models[flag]["config"])
