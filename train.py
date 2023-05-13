import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import wandb
from wandb.keras import WandbCallback


def train():
    DEFAULT_CONFIG = dict(
        epochs=500,
        learning_rate=1e-6,
        batch_size=32,
        img_shape=(224, 224),
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.4,
        es_patience=50,
        valid_split=0.4,
        seed_value=42,
    )

    wandb.init(
        project="wssv-recognition",
        config=DEFAULT_CONFIG,
        group="MobileNetV3Small",
        job_type="train",
    )

    CONFIG = wandb.config

    DATA_DIR = pathlib.Path("augmented_dataset")
    AUTOTUNE = tf.data.AUTOTUNE

    image_count = len(list(DATA_DIR.glob("*/*.jpg")))
    healthy_count = len(list(DATA_DIR.glob("healthy/*.jpg")))
    wssv_count = len(list(DATA_DIR.glob("wssv/*.jpg")))

    print(f"Total number of images: {image_count}")
    print(f"Healthy: {healthy_count}")
    print(f"WSSV: {wssv_count}\n")

    train_set, valid_set = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=CONFIG["valid_split"],
        seed=CONFIG["seed_value"],
        image_size=CONFIG["img_shape"],
        batch_size=CONFIG["batch_size"],
        interpolation="bicubic",
        label_mode="binary",
        subset="both",
    )

    class_names = train_set.class_names
    print(f"\nClass names: {class_names}")

    validation_batches = tf.data.experimental.cardinality(valid_set)
    test_dataset = valid_set.take(validation_batches // 5)
    validation_dataset = valid_set.skip(validation_batches // 5)

    train_dataset = train_set.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    print(
        f"\nNumber of training batches: {tf.data.experimental.cardinality(train_dataset)}"
    )
    print(
        "Number of validation batches: %d"
        % tf.data.experimental.cardinality(validation_dataset)
    )
    print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

    """
    DEFINE BASE MODEL HERE
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(CONFIG["img_height"], CONFIG["img_width"], 3),
        include_top=False,
    )

    """
    CREATE MODEL HERE
    """
    inputs = tf.keras.Input(shape=(CONFIG["img_height"], CONFIG["img_width"], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=CONFIG["dropout_rate"], seed=CONFIG["seed_value"])(
        x
    )
    outputs = tf.keras.layers.Dense(units=2)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(thresholds=0, name="prec"),
            tf.keras.metrics.Recall(thresholds=0, name="recall"),
            tfa.metrics.F1Score(
                num_classes=2, average="weighted", name="f1_score", threshold=0.5
            ),
            tf.keras.metrics.FalseNegatives(name="false_neg"),
        ],
    )

    class_0_weight = (1 / healthy_count) * (image_count / 2.0)
    class_1_weight = (1 / wssv_count) * (image_count / 2.0)
    class_weight = {0: class_0_weight, 1: class_1_weight}

    model.fit(
        train_dataset,
        epochs=CONFIG["epochs"],
        validation_data=validation_dataset,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=CONFIG["es_patience"]),
            WandbCallback(save_model=False),
        ],
    )


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "name": "hyperparameter-search",
        "metric": {"goal": "maximize", "name": "val_f1_score"},
        "parameters": {
            "batch_size": {"distribution": "int-uniform", "values": [4, 8, 16, 32, 64]},
            "epochs": {
                "distribution": "int-uniform",
                "values": [100, 200, 300, 400, 500],
            },
            "learning_rate": {
                "distribution": "uniform",
                "values": [
                    1e-6,
                    1e-5,
                    1e-4,
                    1e-3,
                    1e-2,
                ],
            },
            "dropout_rate": {
                "distribution": "uniform",
                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="wssv-recognition")
    wandb.agent(sweep_id, train, count=10)
