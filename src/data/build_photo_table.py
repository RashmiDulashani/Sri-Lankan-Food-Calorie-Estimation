"""Build photo_table.csv: one row per DISTINCT photo, with plate_id and multi-hot labels."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "metadata"


def main():
    classes = pd.read_csv(META_DIR / "classes.csv")
    class_names = classes[classes.phase == 1].class_name.tolist()

    groups = pd.read_csv(META_DIR / "duplicate_groups.csv")
    if groups.label_conflict.any():
        raise SystemExit("Some photos still have conflicting labels. Fix meals_log.csv first.")

    rows = []
    for _, g in groups.iterrows():
        labels = g.label_sets.split(";")
        row = {
            "photo": g["root"],
            "plate_id": g["root"].split("_")[0],
            "n_copies": int(g.n_files),
        }
        for c in class_names:
            row[c] = int(c in labels)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("photo").reset_index(drop=True)
    df.to_csv(META_DIR / "photo_table.csv", index=False)

    plates = df.groupby("plate_id")[class_names].max()

    print("Distinct photos:", len(df))
    print("Distinct plates:", df.plate_id.nunique())
    print()
    print("Photos per plate:")
    print(df.groupby("plate_id").size().to_string())
    print()
    print("Classes in each plate:")
    print(plates.to_string())
    print()
    print("Plates per class:")
    print(plates.sum().to_string())
    print()
    print("Saved:", META_DIR / "photo_table.csv")


if __name__ == "__main__":
    main()