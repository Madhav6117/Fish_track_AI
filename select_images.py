import os
import shutil

source = "data/images"
destination = "data/selected_images"

os.makedirs(destination, exist_ok=True)

images = sorted([
    f for f in os.listdir(source)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

step = max(1, len(images) // 200)

selected = images[::step]

for image in selected:
    shutil.copy2(
        os.path.join(source, image),
        os.path.join(destination, image)
    )

print(f"Total images found: {len(images)}")
print(f"Images selected: {len(selected)}")
print(f"Saved to: {destination}")