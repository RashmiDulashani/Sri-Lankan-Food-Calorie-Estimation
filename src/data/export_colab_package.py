"""Copy the distinct photos + metadata into one folder and zip it for Google Colab.
Originals in data/raw are only read, never changed."""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"
OUT_DIR = ROOT / "data" / "processed" / "colab_package"


def main():
    folds = pd.read_csv(META_DIR / "folds.csv")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "photos").mkdir(parents=True)
    (OUT_DIR / "metadata").mkdir()

    for name in folds.photo:
        shutil.copy2(RAW_DIR / name, OUT_DIR / "photos" / name)
    for name in ["folds.csv", "photo_table.csv", "classes.csv"]:
        shutil.copy2(META_DIR / name, OUT_DIR / "metadata" / name)

    zip_path = Path(shutil.make_archive(str(OUT_DIR), "zip", root_dir=OUT_DIR))
    print("Photos copied        :", len(folds))
    print("Metadata files copied: 3")
    print("Zip saved            :", zip_path)
    print("Zip size (MB)        :", round(zip_path.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()