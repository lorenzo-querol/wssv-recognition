import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import shutil
import wandb
from wandb.keras import WandbCallback
import os


def train():
    DATA_DIR = pathlib.Path("augmented_dataset")
    AUTOTUNE = tf.data.AUTOTUNE

    DEFAULT_CONFIG = dict(
        epochs=100,
        learning_rate=1e-3,
        batch_size=16,
        img_shape=(224, 224),
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.2,
        es_patience=20,
        valid_split=0.4,
        seed_value=42,
    )

    image_count = len(list(DATA_DIR.glob("*/*.jpg")))
    healthy_count = len(list(DATA_DIR.glob("healthy/*.jpg")))
    wssv_count = len(list(DATA_DIR.glob("wssv/*.jpg")))

    print(f"Total number of images: {image_count}")
    print(f"Healthy: {healthy_count}")
    print(f"WSSV: {wssv_count}\n")

    train_set, valid_set = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=DEFAULT_CONFIG["valid_split"],
        seed=DEFAULT_CONFIG["seed_value"],
        image_size=DEFAULT_CONFIG["img_shape"],
        batch_size=DEFAULT_CONFIG["batch_size"],
        label_mode="binary",
        subset="both",
    )

    class_names = train_set.class_names
    print(f"\nClass names: {class_names}")

    validation_batches = tf.data.experimental.cardinality(valid_set)
    validation_dataset = valid_set.skip(validation_batches // 5)
    test_dataset = valid_set.take(validation_batches // 5)

    train_dataset = train_set.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    num_training_batches = tf.data.experimental.cardinality(train_dataset)
    num_validation_batches = tf.data.experimental.cardinality(validation_dataset)
    num_test_batches = tf.data.experimental.cardinality(test_dataset)

    print(f"\nNumber of training batches: {num_training_batches}")
    print(f"Number of validation batches: {num_validation_batches}")
    print(f"Number of test batches: {num_test_batches}")

    # Define the backbone
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=DEFAULT_CONFIG["input_shape"],
        include_top=False,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=DEFAULT_CONFIG["input_shape"])
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(
        rate=DEFAULT_CONFIG["dropout_rate"], seed=DEFAULT_CONFIG["seed_value"]
    )(x)
    outputs = tf.keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=DEFAULT_CONFIG["learning_rate"]
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.Precision(thresholds=0, name="prec"),
            tf.keras.metrics.Recall(thresholds=0, name="recall"),
            tfa.metrics.F1Score(
                num_classes=1, average="weighted", name="f1_score", threshold=0.5
            ),
            tf.keras.metrics.TruePositives(name="true_pos"),
            tf.keras.metrics.FalseNegatives(name="false_neg"),
        ],
    )

    class_0_weight = (1 / healthy_count) * (image_count / 2.0)
    class_1_weight = (1 / wssv_count) * (image_count / 2.0)
    class_weight = {0: class_0_weight, 1: class_1_weight}

    shutil.rmtree("checkpoints_training", ignore_errors=True)
    checkpoint_path = "checkpoints_training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.fit(
        train_dataset,
        epochs=DEFAULT_CONFIG["epochs"],
        validation_data=validation_dataset,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=DEFAULT_CONFIG["es_patience"]),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
            ),
        ],
    )

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.evaluate(test_dataset, verbose=2)


if __name__ == "__main__":
    train()
