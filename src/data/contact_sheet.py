"""Contact sheet of the distinct photos, with labels. Read-only: changes nothing in data/raw."""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"
FIG_DIR = ROOT / "results" / "figures"

CELL_W, CELL_H, COLS, HEADER = 260, 330, 8, 50


def make_cell(img, name, labels, conflict):
    h, w = img.shape[:2]
    s = min(CELL_W / w, (CELL_H - HEADER) / h)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cell = np.full((CELL_H, CELL_W, 3), 255, np.uint8)
    x0 = (CELL_W - small.shape[1]) // 2
    cell[HEADER:HEADER + small.shape[0], x0:x0 + small.shape[1]] = small
    color = (0, 0, 255) if conflict else (0, 0, 0)
    cv2.putText(cell, name, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(cell, labels[:34], (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return cell


def main():
    groups = pd.read_csv(META_DIR / "duplicate_groups.csv")
    cells = []
    for _, g in groups.iterrows():
        img = cv2.imread(str(RAW_DIR / g["root"]))
        if img is None:
            print("WARNING: unreadable ->", g["root"])
            continue
        cells.append(make_cell(img, g["root"], g["label_sets"], bool(g["label_conflict"])))

    rows = math.ceil(len(cells) / COLS)
    sheet = np.full((rows * CELL_H, COLS * CELL_W, 3), 255, np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, COLS)
        sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "distinct_photos_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("Photos on sheet:", len(cells))
    print("Saved:", out)

    print()
    print("Photos with conflicting labels:")
    bad = groups[groups.label_conflict]
    print(bad[["root", "meals", "label_sets"]].to_string(index=False))


if __name__ == "__main__":
    main()