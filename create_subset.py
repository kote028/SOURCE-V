import os
import shutil
import random

src = "emotion_data/train"
dst = "emotion_subset/train"

os.makedirs(dst, exist_ok=True)

for emotion in os.listdir(src):
    src_folder = os.path.join(src, emotion)
    dst_folder = os.path.join(dst, emotion)
    os.makedirs(dst_folder, exist_ok=True)

    images = os.listdir(src_folder)
    selected = random.sample(images, min(1000, len(images)))

    for img in selected:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dst_folder, img)
        )

print("Subset created ✅")