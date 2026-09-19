import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf

# config
IMG_SIZE = 128
MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES_PATH = "class_names.txt"
PHOTO_PATH = "developer_photo.jpeg"
LOGO_PATH = "logo.png"

DEVELOPER_NAME = "Developed by MOHAMED-AMINE EL-QADERY"
DEVELOPER_TITLE = "Radiology Student at the Higher Institute of\nNursing and Health Techniques Professions\n(ISPITS), Errachidia"

# load model once when the app starts, not every time we predict
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


def predict_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    label = class_names[idx]
    confidence = preds[0][idx] * 100

    return label, confidence, preds[0]


class DiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor MRI Diagnosis")
        self.root.geometry("700x550")

        sidebar = tk.Frame(root, width=200, bg="#1f2937")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        if os.path.exists(LOGO_PATH):
            logo_img = Image.open(LOGO_PATH)
            logo_img.thumbnail((120, 120))
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(sidebar, image=logo_tk, bg="#1f2937")
            logo_label.image = logo_tk  # keep a reference so it doesn't get garbage collected
            logo_label.pack(pady=(20, 10))

        if os.path.exists(PHOTO_PATH):
            photo_img = Image.open(PHOTO_PATH)
            photo_img.thumbnail((140, 140))
            photo_tk = ImageTk.PhotoImage(photo_img)
            photo_label = tk.Label(sidebar, image=photo_tk, bg="#1f2937")
            photo_label.image = photo_tk
            photo_label.pack(pady=10)

        tk.Label(
            sidebar, text=DEVELOPER_NAME, wraplength=180,
            font=("Arial", 10, "bold"), fg="white", bg="#1f2937", justify="center"
        ).pack(pady=(10, 5), padx=10)

        tk.Label(
            sidebar, text=DEVELOPER_TITLE, wraplength=180,
            font=("Arial", 8), fg="#d1d5db", bg="#1f2937", justify="center"
        ).pack(pady=5, padx=10)

        main_area = tk.Frame(root)
        main_area.pack(side="left", fill="both", expand=True)

        self.image_label = tk.Label(main_area, text="No image selected", width=40, height=15, bg="lightgray")
        self.image_label.pack(pady=10)

        tk.Button(main_area, text="Select MRI Image", command=self.select_image, font=("Arial", 12)).pack(pady=5)

        self.result_label = tk.Label(main_area, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)

        self.details_label = tk.Label(main_area, text="", font=("Arial", 10), justify="left")
        self.details_label.pack(pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

        try:
            label, confidence, probs = predict_image(file_path)
            self.result_label.configure(text=f"Diagnosis: {label} ({confidence:.1f}%)")

            details = "\n".join(f"{name}: {p*100:.1f}%" for name, p in zip(class_names, probs))
            self.details_label.configure(text=details)
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DiagnosisApp(root)
    root.mainloop()
