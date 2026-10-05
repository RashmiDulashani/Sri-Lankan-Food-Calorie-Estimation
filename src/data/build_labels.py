"""Build image_labels.csv: one row per photo, multi-hot class labels + marker scale."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"

MARKER_CM = 5.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
default_detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
relaxed_params = cv2.aruco.DetectorParameters()
relaxed_params.adaptiveThreshWinSizeMin = 3
relaxed_params.adaptiveThreshWinSizeMax = 53
relaxed_params.adaptiveThreshWinSizeStep = 4
relaxed_detector = cv2.aruco.ArucoDetector(dictionary, relaxed_params)


def side_length_px(corners):
    """Average side length (in pixels) of the detected marker."""
    pts = np.array(corners[0]).reshape(4, 2)
    return float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))


def marker_px_per_cm(img):
    """Pixels per cm from the ArUco marker, or NaN if no marker is found."""
    corners, ids, _ = default_detector.detectMarkers(img)
    if ids is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        corners, ids, _ = relaxed_detector.detectMarkers(gray)
    if ids is None:
        return np.nan
    return round(side_length_px(corners) / MARKER_CM, 1)


def main():
    classes = pd.read_csv(META_DIR / "classes.csv")
    class_names = classes[classes.phase == 1].class_name.tolist()

    meals = pd.read_csv(META_DIR / "meals_log.csv")
    meal_labels = dict(zip(meals.meal_id, meals.labels))

    for meal_id, labels in meal_labels.items():
        for lab in labels.split(";"):
            if lab not in class_names:
                raise SystemExit(f"Unknown label '{lab}' in {meal_id}. Fix classes.csv or meals_log.csv")

    files = sorted(p for p in RAW_DIR.glob("M*_*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print("Images found:", len(files))

    rows = []
    for i, f in enumerate(files, start=1):
        meal_id = f.stem.split("_")[0]
        if meal_id not in meal_labels:
            print("WARNING: not in meals_log.csv ->", f.name)
            continue
        img = cv2.imread(str(f))
        if img is None:
            print("WARNING: unreadable ->", f.name)
            continue
        labels = meal_labels[meal_id].split(";")
        row = {"image": f.name, "meal_id": meal_id}
        for c in class_names:
            row[c] = int(c in labels)
        row["px_per_cm"] = marker_px_per_cm(img)
        rows.append(row)
        if i % 50 == 0:
            print(f"  processed {i}/{len(files)}")

    labels_df = pd.DataFrame(rows)
    labels_df["marker_detected"] = labels_df.px_per_cm.notna()
    labels_df.to_csv(META_DIR / "image_labels.csv", index=False)

    per_meal = labels_df.groupby("meal_id").size()
    print()
    print("Meals with photos:", len(per_meal), "of", len(meals), "in meals_log.csv")
    print("Meals without exactly 3 photos:", per_meal[per_meal != 3].to_dict())
    print("Meals in log with no photos:", sorted(set(meals.meal_id) - set(per_meal.index)))
    print()
    print("Images per class:")
    print(labels_df[class_names].sum().to_string())
    print()
    print("Marker detected:", int(labels_df.marker_detected.sum()), "/", len(labels_df))


if __name__ == "__main__":
    main()