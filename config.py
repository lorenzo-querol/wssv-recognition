import tensorflow as tf


models = {
    1: {
        "base_model": tf.keras.applications.MobileNetV3Small(
            weights="imagenet",
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg",
        ),
        "model_name": "MobilenetV3small",
        "num_layers_to_freeze": 41,  # 41 223 expanded_conv_13/squeeze_excite/AvgPool
        "config": dict(
            epochs=300,
            learning_rate=1e-6,
            batch_size=64,
            img_shape=(224, 224),
            input_shape=(224, 224, 3),
            num_classes=1,
            dropout_rate=0.2,
            es_patience=30,
            seed_value=42,
        ),
    },
    2: {
        "base_model": tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            input_shape=(224, 224, 3),
            include_top=False,
            pooling="avg",
        ),
        "model_name": "efficientnetv2-b0",
        "num_layers_to_freeze": 71,  # 210 block6e_dwconv2
        "config": dict(
            epochs=300,
            learning_rate=1e-6,
            batch_size=64,
            img_shape=(224, 224),
            input_shape=(224, 224, 3),
            num_classes=1,
            dropout_rate=0.2,
            es_patience=30,
            seed_value=42,
        ),
    },
}
