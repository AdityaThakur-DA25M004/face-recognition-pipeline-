"""
create_duplicate_dataset.py
-----------------------------
Generate duplicate images for evaluation of face authentication system.

Input:
    ./input_images/        # folder with unique clean face images (1 per person)

Output:
    ./evaluation_images/
        person_001/
            img1.jpg
            img2_aug1.jpg
            img2_aug2.jpg
        person_002/
            img1.jpg
            img2_aug1.jpg
            ...

This prepares data compatible with evaluate_duplicates.py.
"""

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "./inference_image_1"           # your source folder
OUTPUT_DIR = "./evaluation_images"     # structured dataset
DUPLICATES_PER_IMAGE = 3               # number of augmented duplicates per image
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# -----------------------------
# Helper Augmentations
# -----------------------------
def adjust_brightness_contrast(image):
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(image).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

def apply_blur(image):
    if random.random() < 0.5:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    else:
        return image.filter(ImageFilter.MedianFilter(size=random.choice([3, 5])))

def add_noise(image):
    img_np = np.array(image)
    noise = np.random.normal(0, random.randint(5, 20), img_np.shape)
    noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def rotate_image(image):
    angle = random.uniform(-10, 10)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)

def jpeg_compression(image):
    quality = random.randint(40, 90)
    temp_path = "temp_compress.jpg"
    image.save(temp_path, "JPEG", quality=quality)
    return Image.open(temp_path)

def random_crop(image):
    width, height = image.size
    crop_ratio = random.uniform(0.9, 1.0)
    new_w, new_h = int(width * crop_ratio), int(height * crop_ratio)
    left = random.randint(0, width - new_w)
    top = random.randint(0, height - new_h)
    return image.crop((left, top, left + new_w, top + new_h)).resize((width, height))

# available augmentations
AUGMENTATIONS = [
    adjust_brightness_contrast,
    apply_blur,
    add_noise,
    rotate_image,
    jpeg_compression,
    random_crop
]

# -----------------------------
# MAIN GENERATION
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n🚀 Generating duplicate dataset from {INPUT_DIR} → {OUTPUT_DIR}\n")

img_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMAGE_EXTS)]

for idx, img_name in enumerate(tqdm(img_list, desc="Processing images")):
    person_id = f"person_{idx+1:03d}"
    person_dir = os.path.join(OUTPUT_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    img_path = os.path.join(INPUT_DIR, img_name)

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Skipping {img_name}: {e}")
        continue

    # save original
    image.save(os.path.join(person_dir, "img1.jpg"))

    # generate duplicate augmented versions
    for i in range(DUPLICATES_PER_IMAGE):
        aug_func = random.choice(AUGMENTATIONS)
        aug_img = aug_func(image)
        aug_img.save(os.path.join(person_dir, f"img_dup{i+1}.jpg"))

print(f"\n✅ Dataset ready for evaluation! Stored in: {OUTPUT_DIR}")
print("Each folder contains the original + duplicate augmented versions.")
