"""
predict.py
Load the trained model and predict the class of a new brain MRI image.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128
MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES_PATH = "class_names.txt"


def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_image(img_path):
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    # Load and preprocess the image
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale"
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    print("-" * 50)
    print(f"Image: {img_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 50)

    print("\nAll class probabilities:")
    for name, prob in zip(class_names, predictions[0]):
        print(f"   {name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    predict_image(img_path)
