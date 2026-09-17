"""
explore_data.py
Explore the Brain Tumor MRI dataset: count images per class
and display sample images from each category.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

# ============================================
# 1. Set dataset path
# ============================================
DATA_DIR = "data"

# Auto-detect if dataset is directly under data/ or under data/archive/
possible_paths = ["data", "data/archive"]
for path in possible_paths:
    if os.path.exists(os.path.join(path, "Training")):
        DATA_DIR = path
        break

TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

print(f"Using dataset path: {DATA_DIR}")
print("-" * 50)

# ============================================
# 2. Count images per class
# ============================================
classes = ["glioma", "meningioma", "pituitary", "notumor"]

def count_images(base_dir, classes):
    counts = {}
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        if os.path.exists(class_path):
            num_images = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            counts[class_name] = num_images
        else:
            counts[class_name] = 0
            print(f"Warning: folder not found: {class_path}")
    return counts

print("Training set image counts:")
train_counts = count_images(TRAIN_DIR, classes)
for class_name, count in train_counts.items():
    print(f"   {class_name}: {count} images")

print()
print("Testing set image counts:")
test_counts = count_images(TEST_DIR, classes)
for class_name, count in test_counts.items():
    print(f"   {class_name}: {count} images")

print("-" * 50)
total = sum(train_counts.values()) + sum(test_counts.values())
print(f"Total images: {total}")
print("-" * 50)

# ============================================
# 3. Display one sample image per class
# ============================================
print("Generating sample image preview...")

fig, axes = plt.subplots(1, len(classes), figsize=(16, 4))

for idx, class_name in enumerate(classes):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.exists(class_path):
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if images:
            img_path = os.path.join(class_path, images[0])
            img = Image.open(img_path)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(class_name)
            axes[idx].axis('off')

plt.tight_layout()
plt.savefig("sample_images.png")
print("Saved preview to sample_images.png")

plt.show()
