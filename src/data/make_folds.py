"""Assign every distinct photo to a CV fold. Folds are made at PLATE level (no leakage)."""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "metadata"

N_SPLITS = 4
SEED = 42


def main():
    classes = pd.read_csv(META_DIR / "classes.csv")
    class_names = classes[classes.phase == 1].class_name.tolist()
    photos = pd.read_csv(META_DIR / "photo_table.csv")

    plates = photos.groupby("plate_id")[class_names].max()
    plates_per_class = plates.sum()

    # each plate is stratified by its RAREST class
    rarest = plates.apply(
        lambda r: min([c for c in class_names if r[c] == 1], key=lambda c: plates_per_class[c]),
        axis=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_of_plate = {}
        for fold, (_, val_idx) in enumerate(skf.split(plates, rarest)):
            for plate_id in plates.index[val_idx]:
                fold_of_plate[plate_id] = fold

    photos["fold"] = photos.plate_id.map(fold_of_plate)
    photos[["photo", "plate_id", "fold"]].to_csv(META_DIR / "folds.csv", index=False)

    # safety check: no plate may appear in two folds
    assert photos.groupby("plate_id").fold.nunique().max() == 1

    print("Plates in each fold:")
    for fold in range(N_SPLITS):
        ids = sorted(plates.index[[fold_of_plate[p] == fold for p in plates.index]])
        val = plates.loc[ids]
        train = plates.drop(ids)
        missing_val = [c for c in class_names if val[c].sum() == 0]
        missing_train = [c for c in class_names if train[c].sum() == 0]
        print(f"  fold {fold}: {ids}")
        print(f"     photos: {int((photos.fold == fold).sum())}, plates per class in val: {val.sum().to_dict()}")
        print(f"     classes missing in val: {missing_val} | missing in train: {missing_train}")
    print()
    print("Saved:", META_DIR / "folds.csv")


if __name__ == "__main__":
    main()