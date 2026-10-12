"""Dataset and image transforms for the Sri Lankan food photos."""

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    """Resize + normalise for everyone; augmentation for TRAINING photos only."""
    steps = [transforms.Resize((224, 224))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def load_table(meta_dir):
    """Return (table with labels + fold for every distinct photo, list of class names)."""
    classes = pd.read_csv(f"{meta_dir}/classes.csv")
    class_names = classes[classes.phase == 1].class_name.tolist()
    photos = pd.read_csv(f"{meta_dir}/photo_table.csv")
    folds = pd.read_csv(f"{meta_dir}/folds.csv")
    return photos.merge(folds[["photo", "fold"]], on="photo"), class_names


class FoodDataset(Dataset):
    def __init__(self, table, class_names, photo_dir, transform):
        self.table = table.reset_index(drop=True)
        self.class_names = class_names
        self.photo_dir = photo_dir
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        img = Image.open(f"{self.photo_dir}/{row.photo}")
        img = ImageOps.exif_transpose(img).convert("RGB")
        y = torch.tensor([row[c] for c in self.class_names], dtype=torch.float32)
        return self.transform(img), y