"""
Image preprocessing pipeline for the Sri Lankan Food Recognition
and Calorie Estimation project.

Steps: quality filtering -> resize -> normalise -> augment -> reference-object calibration

Usage:
    python preprocess.py --input data/raw --output data/processed
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
TARGET_SIZE = (224, 224)

BLUR_THRESHOLD = 100.0  # Variance of Laplacian below this is treated as "too blurry"


def is_blurry(image, threshold: float = BLUR_THRESHOLD) -> bool:
    """Step 1: quality filtering — flags low-sharpness images using the
    variance-of-Laplacian method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def resize_image(image, size=TARGET_SIZE):
    """Step 2: resize to the EfficientNet-B0 input size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalise_image(image):
    """Step 3: scale pixel values using ImageNet mean/std."""
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def get_augmentation_pipeline():
    """Step 4: augmentation — applied to training images only."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])


def estimate_portion_scale(reference_object_area_px: float, reference_object_real_cm2: float = 4.0):
    """Step 5: reference-object calibration.

    Converts pixel measurements to an approximate real-world scale using a
    reference object of known physical size (e.g. a coin) placed beside the
    plate. Returns cm^2 per pixel.
    """
    if reference_object_area_px <= 0:
        raise ValueError("Reference object area must be a positive number of pixels.")
    return reference_object_real_cm2 / reference_object_area_px


def process_folder(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    augment = get_augmentation_pipeline()
    kept, skipped = 0, 0

    for class_folder in input_path.iterdir():
        if not class_folder.is_dir():
            continue
        out_class_folder = output_path / class_folder.name
        out_class_folder.mkdir(parents=True, exist_ok=True)

        for img_file in class_folder.glob("*.*"):
            image = cv2.imread(str(img_file))
            if image is None:
                skipped += 1
                continue

            if is_blurry(image):
                skipped += 1
                continue

            image = resize_image(image)
            augmented = augment(image=image)["image"]

            out_file = out_class_folder / img_file.name
            cv2.imwrite(str(out_file), augmented)
            kept += 1

    print(f"Preprocessing complete: {kept} images kept, {skipped} images skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sri Lankan food images.")
    parser.add_argument("--input", required=True, help="Path to raw image folder (one subfolder per class)")
    parser.add_argument("--output", required=True, help="Path to save processed images")
    args = parser.parse_args()
    process_folder(args.input, args.output)