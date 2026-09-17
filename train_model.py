from curses import version


py -3.12 --version"""
train_model.py
Train a CNN to classify brain MRI images into 4 categories:
glioma, meningioma, pituitary, notumor.

Optimized for weak hardware (CPU or low-end GPU):
- Small image size (128x128)
- Lightweight CNN architecture
- Limited number of epochs
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================
# 1. Settings
# ============================================
IMG_SIZE = 128          # Smaller = faster training, less accuracy
BATCH_SIZE = 16          # Smaller = uses less RAM
EPOCHS = 10              # Increase later for better accuracy once it works

DATA_DIR = "data"
possible_paths = ["data", "data/archive"]
for path in possible_paths:
    if os.path.exists(os.path.join(path, "Training")):
        DATA_DIR = path
        break

TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

print(f"Using dataset path: {DATA_DIR}")

# ============================================
# 2. Load datasets
# ============================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Save class names for later use in predict.py
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# Normalize pixel values (0-255 -> 0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================
# 3. Build a lightweight CNN model
# ============================================
num_classes = len(class_names)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 4. Train the model
# ============================================
print("Starting training... this may take a while on CPU.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ============================================
# 5. Save the trained model
# ============================================
model.save("brain_tumor_model.keras")
print("Model saved as brain_tumor_model.keras")

# ============================================
# 6. Print final accuracy
# ============================================
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final training accuracy: {final_acc:.2%}")
print(f"Final validation accuracy: {final_val_acc:.2%}")
