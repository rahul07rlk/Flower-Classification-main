import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape: tuple, num_classes: int):
    """
    Builds and compiles a transfer learning model using MobileNetV2 as the base.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        model: A compiled Keras model.
    """
    # Load a pre-trained MobileNetV2 model, excluding the top classification layers.
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Create a new model on top of the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
