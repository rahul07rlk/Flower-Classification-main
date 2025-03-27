import tensorflow as tf


def load_datasets(train_dir: str, valid_dir: str, img_size: tuple, batch_size: int, seed: int = 123):
    """
    Loads training and validation datasets using a directory structure.

    Args:
        train_dir (str): Path to training directory.
        valid_dir (str): Path to validation directory.
        img_size (tuple): Desired image size (height, width).
        batch_size (int): Batch size.
        seed (int): Random seed for shuffling.

    Returns:
        train_ds: tf.data.Dataset for training.
        valid_ds: tf.data.Dataset for validation.
        class_names: List of class names.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names  # Assumes both directories have identical classes
    return train_ds, valid_ds, class_names
