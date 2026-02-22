"""
Training script for Jaguar Re-ID. Run as notebook cells or python train.py.
"""
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    JaguarTrainDataset,
    JaguarTestDataset,
    get_train_transform,
    get_test_transform,
    get_balanced_sampler,
)
from models import ReIDModel, ArcFace
from metrics import identity_balanced_map


# ---------------------------------------------------------------------------
# Config (override for Kaggle paths)
# ---------------------------------------------------------------------------

def get_config(kaggle=True):
    return {
        "train_csv": "/kaggle/input/jaguar-re-id/train.csv" if kaggle else "train.csv",
        "train_dir": "/kaggle/input/jaguar-re-id/train/train" if kaggle else "train/train",
        "test_dir": "/kaggle/input/jaguar-re-id/test/test" if kaggle else "test/test",
        "test_csv": "/kaggle/input/jaguar-re-id/test.csv" if kaggle else "test.csv",
        "img_size": 224,
        "embedding_dim": 512,
        "backbone": "resnet50",
        "batch_size": 32,
        "num_workers": 2,
        "epochs": 20,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "use_balanced_sampler": True,
        "use_strong_aug": True,
        "use_alpha_mask": False,
        "mask_dir": None,
        "grad_clip": 1.0,
        "save_path": "/kaggle/working/best_model.pt",
    }


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, arcface_head, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    arcface_head.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        emb = model(images)
        logits = arcface_head(emb, labels)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface_head.parameters()), grad_clip
            )
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        pbar.set_postfix(loss=loss.item())
    return total_loss / n


# ---------------------------------------------------------------------------
# Build embedding dict (for validation or test)
# ---------------------------------------------------------------------------

def build_embedding_dict(model, loader, device, normalize=True):
    model.eval()
    embedding_dict = {}
    with torch.no_grad():
        for images, names in tqdm(loader, desc="Embedding", leave=False):
            images = images.to(device)
            emb = model(images)
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)
            for name, e in zip(names, emb.cpu()):
                embedding_dict[name] = e
    return embedding_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", default=True, help="Use Kaggle paths")
    parser.add_argument("--no-kaggle", action="store_false", dest="kaggle")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50")
    args = parser.parse_args()

    cfg = get_config(kaggle=args.kaggle)
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["backbone"] = args.backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = JaguarTrainDataset(
        cfg["train_csv"],
        cfg["train_dir"],
        transform=get_train_transform(cfg["img_size"], use_strong_aug=cfg["use_strong_aug"]),
        use_alpha_mask=cfg["use_alpha_mask"],
        mask_dir=cfg["mask_dir"],
    )
    sampler = get_balanced_sampler(train_dataset) if cfg["use_balanced_sampler"] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    num_classes = train_dataset.num_classes
    model = ReIDModel(
        embedding_dim=cfg["embedding_dim"],
        backbone=cfg["backbone"],
        pretrained=True,
    ).to(device)
    arcface_head = ArcFace(
        embedding_dim=cfg["embedding_dim"],
        num_classes=num_classes,
        s=30.0,
        m=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_loss = float("inf")
    for epoch in range(cfg["epochs"]):
        loss = train_epoch(
            model, arcface_head, train_loader, criterion, optimizer, device, cfg["grad_clip"]
        )
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg['epochs']}  Loss: {loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")
        if loss < best_loss:
            best_loss = loss
            state = {
                "model": model.state_dict(),
                "arcface": arcface_head.state_dict(),
                "num_classes": num_classes,
                "embedding_dim": cfg["embedding_dim"],
                "label_map": train_dataset.label_map,
            }
            torch.save(state, cfg["save_path"])
            print(f"  -> Saved to {cfg['save_path']}")

    print("Training done. Run inference to build embedding_dict and create submission.csv.")


if __name__ == "__main__":
    main()
