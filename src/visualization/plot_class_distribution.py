from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files only; no pop-up window needed

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "metadata"
FIG_DIR = ROOT / "results" / "figures"


def main():
    classes = pd.read_csv(META_DIR / "classes.csv")
    class_names = classes[classes.phase == 1].class_name.tolist()
    df = pd.read_csv(META_DIR / "image_labels.csv")

    images_per_class = df[class_names].sum()
    meals = df.groupby("meal_id")[class_names].max()
    meals_per_class = meals.sum()
    items_per_plate = meals.sum(axis=1)

    print("Images per class:")
    print(images_per_class.to_string())
    print()
    print("Meals per class:")
    print(meals_per_class.to_string())
    print()
    print("Food items per plate (number of meals):")
    print(items_per_plate.value_counts().sort_index().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], images_per_class, "Images per class"),
        (axes[1], meals_per_class, "Meals (plates) per class"),
    ]
    for ax, counts, title in panels:
        bars = ax.bar(class_names, counts.values, color="#2E5597")
        ax.bar_label(bars)
        ax.set_title(title)
        ax.set_xlabel("Food class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "class_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()