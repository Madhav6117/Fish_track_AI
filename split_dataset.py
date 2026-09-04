import os
import shutil
import random

source_images = "dataset/images"
source_labels = "dataset/labels"

train_images = "dataset/images/train"
val_images = "dataset/images/val"

train_labels = "dataset/labels/train"
val_labels = "dataset/labels/val"

os.makedirs(train_images, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

images = [
    f for f in os.listdir(source_images)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)

split = int(len(images) * 0.8)

train_set = images[:split]
val_set = images[split:]

for image in train_set:
    name = os.path.splitext(image)[0]

    shutil.copy2(
        os.path.join(source_images, image),
        os.path.join(train_images, image)
    )

    shutil.copy2(
        os.path.join(source_labels, name + ".txt"),
        os.path.join(train_labels, name + ".txt")
    )

for image in val_set:
    name = os.path.splitext(image)[0]

    shutil.copy2(
        os.path.join(source_images, image),
        os.path.join(val_images, image)
    )

    shutil.copy2(
        os.path.join(source_labels, name + ".txt"),
        os.path.join(val_labels, name + ".txt")
    )

print("Dataset split complete!")
print(f"Training images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")