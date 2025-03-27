# data_loader.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_datasets(data_dir: str, img_size: tuple, batch_size: int, seed: int):
    """
    Loads the training and validation datasets from the specified directory.
    Assumes the data is organized in subdirectories for each class.

    Returns:
        train_ds, val_ds: TensorFlow Dataset objects for training and validation.
    """
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    return train_ds, val_ds
