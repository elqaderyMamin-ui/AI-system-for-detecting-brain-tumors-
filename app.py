"""
app.py
Simple desktop GUI for brain tumor diagnosis.
Lets the user pick an MRI image and shows the predicted class.
"""

import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf

IMG_SIZE = 128
MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES_PATH = "class_names.txt"

# ============================================
# Load model and class names once at startup
# ============================================
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


def predict_image(img_path):
    img = Image.open(img_path).convert("L")  # convert to grayscale
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, H, W, 1)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence, predictions[0]


# ============================================
# GUI
# ============================================
class DiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor MRI Diagnosis")
        self.root.geometry("450x550")

        self.image_label = tk.Label(root, text="No image selected", width=40, height=15, bg="lightgray")
        self.image_label.pack(pady=10)

        select_btn = tk.Button(root, text="Select MRI Image", command=self.select_image, font=("Arial", 12))
        select_btn.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

        self.details_label = tk.Label(root, text="", font=("Arial", 10), justify="left")
        self.details_label.pack(pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        # Show the selected image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

        # Run prediction
        try:
            predicted_class, confidence, all_probs = predict_image(file_path)
            self.result_label.configure(
                text=f"Diagnosis: {predicted_class} ({confidence:.1f}%)"
            )

            details = "\n".join(
                f"{name}: {prob*100:.1f}%" for name, prob in zip(class_names, all_probs)
            )
            self.details_label.configure(text=details)
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DiagnosisApp(root)
    root.mainloop()
