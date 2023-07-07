import os
import shutil
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from utils import fine_tune
from config import models

AUTOTUNE = tf.data.AUTOTUNE


def compute_class_weights(data_dir: str):
    dir_class1 = os.path.join(data_dir, "healthy")
    dir_class2 = os.path.join(data_dir, "wssv")

    # Count the number of images in each directory
    count_class1 = len(os.listdir(dir_class1))
    count_class2 = len(os.listdir(dir_class2))

    # Total number of images
    total = count_class1 + count_class2

    # Calculate class weights
    weight_for_class1 = (1 / count_class1) * (total) / 2.0
    weight_for_class2 = (1 / count_class2) * (total) / 2.0

    class_weight = {0: weight_for_class1, 1: weight_for_class2}

    return class_weight


def fine_tune_model(flag: int, data_dir: str, config: dict):
    """
    Train the model using K-fold cross-validation. The model weights are saved in the checkpoints folder.
    The training logs are saved in the results folder. The model with the lowest validation loss is saved.

    Inputs:
    - flag: An integer to select the base model and model name
    - data_dir: Path to the dataset directory
    """

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    train_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=config["seed_value"],
        image_size=config["img_shape"],
        batch_size=config["batch_size"],
        label_mode="binary",
    ).prefetch(buffer_size=AUTOTUNE)

    valid_set = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        seed=config["seed_value"],
        image_size=config["img_shape"],
        batch_size=config["batch_size"],
        label_mode="binary",
    ).prefetch(buffer_size=AUTOTUNE)

    base_model = models[flag]["base_model"]
    model_name = models[flag]["model_name"]
    num_layers_to_freeze = models[flag]["num_layers_to_freeze"]

    print(
        f"Training {model_name} for {config['epochs']} epochs with {config['batch_size']} batch size, dropout rate of {config['dropout_rate']}, and learning rate of {config['learning_rate']}..."
    )
    print(f"Fine tuning {num_layers_to_freeze} layers...")

    # Create a new model instance
    model = fine_tune(base_model, config, num_layers_to_freeze)

    # Define checkpoint path and checkpoint callback
    path = f"checkpoints/fine-tune/{model_name}"
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
            f"results/fine-tune/{model_name}.csv", separator=",", append=False
        ),
    ]

    # Compute class weights
    class_weight = compute_class_weights(train_dir)

    # Fit the model on the train set and evaluate on the validation set
    model.fit(
        train_set,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        class_weight=class_weight,
        validation_data=valid_set,
        verbose=1,
        callbacks=callbacks,
    )

    # Save the model weights
    path = f"checkpoints/fine-tune/{model_name}/best_weights.ckpt"
    model.save_weights(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flag",
        type=int,
        default=1,
        help="An integer to select the model. (1) MobilenetV3Small, (2) EfficientNetV2B0, (3) MobilenetV3Large",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the dataset directory",
    )

    args = parser.parse_args()
    flag = args.flag
    data_dir = args.data_dir

    fine_tune_model(flag, data_dir, models[flag]["config"])
