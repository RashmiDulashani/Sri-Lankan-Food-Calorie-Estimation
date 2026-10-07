"""Train a frozen-backbone classifier (ResNet-50 or EfficientNet-B0) on plate-level CV folds.

Run from the repository root, for example on Google Colab:
    python -m src.training.train --model efficientnet_b0 --photo-dir /content/data/photos --meta-dir /content/data/metadata
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

from src.data.dataset import FoodDataset, get_transforms, load_table


def build_model(name, num_classes, device):
    """Pretrained backbone (frozen) + a new trainable last layer."""
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.eval()  # backbone is frozen, so keep BatchNorm / Dropout fixed
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            loss = criterion(model(x), y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / count


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.sigmoid(model(x.to(device))).cpu())
    return torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser(description="Train a model on all CV folds.")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--photo-dir", required=True)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--out-dir", default="results/metrics")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| model:", args.model)

    data, class_names = load_table(args.meta_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_rows, pred_frames = [], []
    for fold in sorted(data.fold.unique()):
        torch.manual_seed(args.seed)
        train_table = data[data.fold != fold]
        val_table = data[data.fold == fold].reset_index(drop=True)
        assert not set(train_table.plate_id) & set(val_table.plate_id), "plate leakage!"

        train_loader = DataLoader(
            FoodDataset(train_table, class_names, args.photo_dir, get_transforms(True)),
            batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(
            FoodDataset(val_table, class_names, args.photo_dir, get_transforms(False)),
            batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model, len(class_names), device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(trainable, lr=args.lr)

        if fold == 0:
            n_train = sum(p.numel() for p in trainable)
            n_all = sum(p.numel() for p in model.parameters())
            print(f"trainable parameters: {n_train:,} of {n_all:,}")

        print(f"\nfold {fold}: train photos {len(train_table)}, val photos {len(val_table)}")
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)
            log_rows.append({"fold": fold, "epoch": epoch,
                             "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch:02d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")

        probs = predict(model, val_loader, device)
        frame = val_table[["photo", "plate_id", "fold"]].copy()
        for j, c in enumerate(class_names):
            frame[f"true_{c}"] = val_table[c].values
            frame[f"prob_{c}"] = probs[:, j].round(4)
        pred_frames.append(frame)

    pd.DataFrame(log_rows).to_csv(out_dir / f"{args.model}_training_log.csv", index=False)
    pd.concat(pred_frames).to_csv(out_dir / f"{args.model}_predictions.csv", index=False)
    print("\nSaved to", out_dir)


if __name__ == "__main__":
    main()