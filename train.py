# train.py

import tensorflow as tf
from tensorflow.keras import callbacks
from data_loader import load_datasets
from model_builder import build_model

# -------------------------------
# Configuration and Hyperparameters
# -------------------------------
DATA_DIR = 'data/flowers'  # Update to the location of your dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # Update based on your dataset (number of flower types)
EPOCHS = 50
SEED = 123


def train_model():
    # Load datasets
    train_ds, val_ds = load_datasets(DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED)

    # Optimize data pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    model = build_model(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
    model.summary()

    # Setup callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    )
    earlystop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nValidation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # Save the final model
    model.save("flower_classifier_model.h5")
    print("Final model saved successfully!")


if __name__ == "__main__":
    # Optional: Configure GPU memory growth if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_model()
