"""Summarise duplicate photos from quality_report.csv. Does NOT delete anything."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "metadata"


def main():
    report = pd.read_csv(META_DIR / "quality_report.csv").fillna("")
    meals = pd.read_csv(META_DIR / "meals_log.csv")
    labels = dict(zip(meals.meal_id, meals.labels))

    # every copy points to the first file with identical bytes
    report["root"] = report.apply(
        lambda r: r.exact_duplicate_of if r.exact_duplicate_of else r.image, axis=1
    )
    report["labels"] = report.meal_id.map(labels)

    groups = report.groupby("root").agg(
        n_files=("image", "size"),
        n_meals=("meal_id", "nunique"),
        meals=("meal_id", lambda s: ";".join(sorted(set(s)))),
        label_sets=("labels", lambda s: " | ".join(sorted(set(s)))),
    )
    groups["label_conflict"] = groups.label_sets.str.contains(r"\|")
    groups.to_csv(META_DIR / "duplicate_groups.csv")

    print("Total image files        :", len(report))
    print("Distinct photos (unique) :", report.root.nunique())
    print("Copies (extra files)     :", len(report) - report.root.nunique())
    print()
    print("Photos used in more than 1 meal ID :", int((groups.n_meals > 1).sum()))
    print("Photos whose meals have DIFFERENT labels:", int(groups.label_conflict.sum()))
    print()
    print("Meals that share at least one photo with another meal:")
    shared = groups[groups.n_meals > 1].meals.str.split(";").explode().nunique()
    print("  ", shared, "of", report.meal_id.nunique(), "meals")
    print()
    print("Unique photos per label set:")
    first = report.drop_duplicates("root")
    print(first.labels.value_counts().to_string())
    print()
    print("Saved:", META_DIR / "duplicate_groups.csv")


if __name__ == "__main__":
    main()