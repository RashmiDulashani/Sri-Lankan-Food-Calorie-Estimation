"""Quality check for raw photos. Writes a report; does NOT delete or change any image."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CHECK_SIZE = 1024       # longest side (px) used when measuring blur
BLUR_FRACTION = 0.5     # flag if blur score < 50% of the median score
DARK_LIMIT = 60         # flag if mean brightness (0-255) is below this
BRIGHT_LIMIT = 200      # flag if mean brightness (0-255) is above this
NEAR_DUP_BITS = 4       # hashes differing by <= 4 of 64 bits = near duplicate


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def blur_score(gray):
    """Variance of the Laplacian after resizing: higher = sharper."""
    h, w = gray.shape
    scale = CHECK_SIZE / max(h, w)
    small = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def dhash(gray):
    """64-bit 'fingerprint' of the picture's look (same look -> similar bits)."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def main():
    files = sorted(p for p in RAW_DIR.glob("M*_*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print("Images found:", len(files))

    rows, hashes, first_seen = [], [], {}
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print("WARNING: unreadable ->", f.name)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        digest = md5_of(f)
        rows.append({
            "image": f.name,
            "meal_id": f.stem.split("_")[0],
            "width": img.shape[1],
            "height": img.shape[0],
            "blur_score": round(blur_score(gray), 1),
            "brightness": round(float(gray.mean()), 1),
            "exact_duplicate_of": first_seen.get(digest, ""),
        })
        first_seen.setdefault(digest, f.name)
        hashes.append(dhash(gray))

    df = pd.DataFrame(rows)

    # near duplicates: similar-looking photos that belong to DIFFERENT meals
    H = np.array(hashes)
    dist = (H[:, None, :] != H[None, :, :]).sum(axis=2)
    near = [""] * len(df)
    for i in range(len(df)):
        for j in range(i):
            if df.meal_id[i] != df.meal_id[j] and dist[i, j] <= NEAR_DUP_BITS:
                near[i] = df.image[j]
                break
    df["near_duplicate_of"] = near

    median_blur = df.blur_score.median()
    flags = []
    for _, r in df.iterrows():
        reasons = []
        if r.blur_score < BLUR_FRACTION * median_blur:
            reasons.append("blurry?")
        if r.brightness < DARK_LIMIT:
            reasons.append("dark")
        if r.brightness > BRIGHT_LIMIT:
            reasons.append("bright")
        if r.exact_duplicate_of:
            reasons.append("exact_duplicate")
        if r.near_duplicate_of:
            reasons.append("near_duplicate")
        flags.append(";".join(reasons))
    df["flag"] = flags

    out = META_DIR / "quality_report.csv"
    df.to_csv(out, index=False)

    print()
    print("Blur score  min / median / max:",
          df.blur_score.min(), "/", median_blur, "/", df.blur_score.max())
    print("Brightness  min / median / max:",
          df.brightness.min(), "/", df.brightness.median(), "/", df.brightness.max())
    print()
    print("10 lowest blur scores:")
    print(df.nsmallest(10, "blur_score")[["image", "blur_score", "brightness"]].to_string(index=False))
    print()
    flagged = df[df.flag != ""]
    print("Flagged images:", len(flagged), "of", len(df))
    if len(flagged):
        print(flagged[["image", "blur_score", "brightness", "flag"]].to_string(index=False))
    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()