import os
import tensorflow as tf
import numpy as np
from data_loader import load_datasets
from model_builder import build_model
import PIL.Image

# -------------------------------
# Configuration and Hyperparameters
# -------------------------------
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
SAMPLE_DIR = 'data/sample_images'  # Folder with sample images for evaluation
IMG_SIZE = (224, 224)  # Change as needed
BATCH_SIZE = 32
EPOCHS = 50


def predict_img(img, learn, df):
    """
    Get prediction from the trained model.
    Returns:
        class_name: The human-readable name.
        pred_idx: The predicted class index (integer).
        probability: The prediction probability (percentage).
        details: Additional details (if any).
    """
    try:
        img_array = np.asarray(img)
        pred_class, pred_idx, probs = learn.predict(img_array)
        probability = round(np.max(np.array(probs)) * 100, 2)
        # Using the CSV data (df) to get the flower name and details:
        class_name = get_name(df, int(pred_idx))
        details = get_details(df, int(pred_idx))
        return class_name, int(pred_idx), probability, details
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", -1, 0, None


# -----
# We assume that data_loader.py provides get_name() and get_details() functions.
# If not, define them here or import from a module.
from data_loader import get_name, get_details


def evaluate_sample_images(sample_dir, learn, df):
    """
    Evaluates sample images in sample_dir.
    Assumes that each image file is named with the ground truth label as the first part,
    e.g. "1_image1.jpg" means the true label is 1.
    Prints individual predictions and overall accuracy.
    """
    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = 0
    correct = 0
    print("\nEvaluating sample images:")
    for file in files:
        try:
            img = PIL.Image.open(file)
            pred_class, pred_idx, prob, _ = predict_img(img, learn, df)
            # Parse ground truth from filename: assume the filename starts with the label (e.g., "1_...")
            basename = os.path.basename(file)
            gt_str = basename.split('_')[0]
            gt = int(gt_str)
            total += 1
            if gt == pred_idx:
                correct += 1
            print(f"File: {basename} | GT: {gt} | Pred: {pred_idx} ({pred_class}) | Prob: {prob}%")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if total > 0:
        accuracy = correct / total * 100
        print(f"\nSample images accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("No sample images found.")


def main():
    # Load training and validation datasets
    train_ds, valid_ds, class_names = load_datasets(TRAIN_DIR, VALID_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Optimize datasets with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    model = build_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    model.summary()

    # Set up callbacks: EarlyStopping and ModelCheckpoint
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Train the model
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=callbacks_list)

    # Evaluate on the validation dataset
    loss, acc = model.evaluate(valid_ds)
    print(f"\nValidation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")

    # Save the final model
    model.save("flower_classifier_model.h5")
    print("Model saved as flower_classifier_model.h5")

    # --- Evaluate sample images ---
    # For evaluating sample images, we need the same CSV data that provides names and details.
    # Assume you have a CSV file (e.g., Flowers.csv) and functions get_name() and get_details() in data_loader.py.
    # Here, we load the CSV data (this could be cached in data_loader.py as well).
    import pandas as pd
    try:
        df = pd.read_csv("Flowers.csv", index_col=['Index'])
    except Exception as e:
        print(f"Error loading Flowers.csv: {e}")
        df = None

    if df is not None:
        evaluate_sample_images(SAMPLE_DIR, model, df)
    else:
        print("Skipping sample images evaluation due to missing CSV data.")


if __name__ == '__main__':
    main()
