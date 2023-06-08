import tensorflow as tf
import tensorflow_addons as tfa
import os
import glob
import wandb
from wandb.keras import WandbCallback
import shutil

DATA_DIR = "dataset-augmented"
AUTOTUNE = tf.data.AUTOTUNE
CONFIG = dict(
    epochs=100,
    learning_rate=1e-4,
    batch_size=8,
    img_shape=(224, 224),
    input_shape=(224, 224, 3),
    num_classes=2,
    dropout_rate=0.6,
    es_patience=10,
    seed_value=42,
)


def train():
    wandb.init(
        project="wssv-recognition",
        config=CONFIG,
        group="efficientnetv2b0",
        job_type="sweep",
    )

    config = wandb.config
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    img_shape = config.img_shape
    input_shape = config.input_shape
    dropout_rate = config.dropout_rate
    es_patience = config.es_patience
    seed_value = config.seed_value

    image_count = len(glob.glob(f"{DATA_DIR}/*/*/*.jpg"))
    healthy_count = len(glob.glob(f"{DATA_DIR}/train/healthy/*.jpg"))
    wssv_count = len(glob.glob(f"{DATA_DIR}/train/wssv/*.jpg"))

    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    train_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=seed_value,
        image_size=img_shape,
        batch_size=batch_size,
        label_mode="categorical",
    ).prefetch(buffer_size=AUTOTUNE)

    valid_set = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        seed=seed_value,
        image_size=img_shape,
        batch_size=batch_size,
        label_mode="categorical",
    ).prefetch(buffer_size=AUTOTUNE)

    test_set = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=seed_value,
        image_size=img_shape,
        batch_size=batch_size,
        label_mode="categorical",
    ).prefetch(buffer_size=AUTOTUNE)

    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        classes=2,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate, seed=seed_value)(x)
    outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tfa.metrics.F1Score(
                num_classes=2,
                average="weighted",
                name="f1_score",
                threshold=0.5,
            ),
            tf.keras.metrics.FalseNegatives(name="false_negatives"),
            tf.keras.metrics.TruePositives(name="true_positives"),
        ],
    )

    class_0_weight = (1 / healthy_count) * (image_count / 2.0)
    class_1_weight = (1 / wssv_count) * (image_count / 2.0)
    class_weight = {0: class_0_weight, 1: class_1_weight}

    model.fit(
        train_set,
        epochs=epochs,
        validation_data=valid_set,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=es_patience),
            WandbCallback(save_model=False),
        ],
    )


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize",
    },
    "parameters": {
        "optimizer": {"value": "adam"},
        "epochs": {"value": 100},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "batch_size": {"values": [4, 8, 16, 32]},
        "dropout_rate": {"values": [0.3, 0.4, 0.5, 0.6]},
    },
}

wandb.login()
sweep_id = wandb.sweep(SWEEP_CONFIG, project="wssv-recognition")
wandb.agent(sweep_id, train, count=50)
