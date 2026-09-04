import os
import shutil

source_images = "data/selected_images"
source_labels = "labels"

dataset_images = "dataset/images"
dataset_labels = "dataset/labels"

os.makedirs(dataset_images, exist_ok=True)
os.makedirs(dataset_labels, exist_ok=True)

image_files = {
    os.path.splitext(f)[0]: f
    for f in os.listdir(source_images)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

copied = 0
missing = 0

for label_file in os.listdir(source_labels):

    if not label_file.endswith(".txt"):
        continue

    # Label Studio names labels like:
    # randomid-fish_0905.txt
    label_name = os.path.splitext(label_file)[0]
    image_name = label_name.split("-", 1)[-1]

    if image_name in image_files:
        original_image = image_files[image_name]

        shutil.copy2(
            os.path.join(source_images, original_image),
            os.path.join(dataset_images, original_image)
        )

        shutil.copy2(
            os.path.join(source_labels, label_file),
            os.path.join(dataset_labels, image_name + ".txt")
        )

        copied += 1
    else:
        missing += 1

print("Dataset preparation complete!")
print(f"Images copied: {copied}")
print(f"Labels copied: {copied}")
print(f"Missing images: {missing}")