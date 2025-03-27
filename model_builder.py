# model_builder.py

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape: tuple, num_classes: int):
    """
    Builds and compiles a CNN model with integrated data augmentation.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        num_classes: Number of output classes.

    Returns:
        model: A compiled Keras model.
    """
    # Data augmentation block
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Construct the CNN model
    model = models.Sequential(name="Flower_Classifier")
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(data_augmentation)
    model.add(layers.Rescaling(1. / 255))

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
