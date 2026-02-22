"""
Jaguar Re-ID Datasets with optional alpha mask, augmentations, and balanced sampling.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform(img_size=224, use_strong_aug=True):
    """ReID-style transforms: normalize, color jitter, random erasing to reduce background reliance."""
    base = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if not use_strong_aug:
        return T.Compose(base)

    # Strong aug: color jitter + random grayscale (encourages learning spots, not background)
    train_transforms = [
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]
    return T.Compose(train_transforms)


def get_test_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Optional: use alpha channel to mask background (reduce spurious correlations)
# ---------------------------------------------------------------------------

def apply_alpha_mask(rgb: np.ndarray, alpha: np.ndarray, fill_color=(114, 114, 114)):
    """Composite jaguar onto neutral gray background using alpha. Reduces background cues."""
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    alpha = np.expand_dims(alpha, axis=2)
    rgb = rgb.astype(np.float32)
    fill = np.array(fill_color, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * alpha + fill * (1 - alpha)
    return out.astype(np.uint8)


def load_image_with_optional_mask(img_path: str, use_mask: bool, mask_dir: str = None):
    """Load RGB and optionally composite with alpha mask."""
    img = np.array(Image.open(img_path).convert("RGB"))
    if not use_mask or mask_dir is None:
        return img

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    # Common pattern: mask in same name with _mask.png or in separate folder
    mask_path = os.path.join(mask_dir, name + "_mask" + ext)
    if not os.path.isfile(mask_path):
        mask_path = os.path.join(mask_dir, base)
    if os.path.isfile(mask_path):
        mask_img = Image.open(mask_path)
        if mask_img.mode == "RGBA":
            alpha = np.array(mask_img.split()[-1])
        else:
            alpha = np.array(mask_img.convert("L"))
        img = apply_alpha_mask(img, alpha)
    return img


# ---------------------------------------------------------------------------
# Train Dataset
# ---------------------------------------------------------------------------

class JaguarTrainDataset(Dataset):
    """Train set: (image, label). Supports optional alpha mask and class map."""

    def __init__(self, csv_path: str, image_dir: str, transform=None,
                 use_alpha_mask: bool = False, mask_dir: str = None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir.rstrip("/")
        self.transform = transform
        self.use_alpha_mask = use_alpha_mask
        self.mask_dir = mask_dir

        # Build label map: identity -> 0, 1, 2, ...
        self.identities = sorted(self.df["id"].unique()) if "id" in self.df.columns else sorted(
            self.df["label"].unique()
        )
        self.label_map = {name: i for i, name in enumerate(self.identities)}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Support columns: image, image_name, file, etc.
        img_name = row.get("image", row.get("image_name", row.get("file", "image")))
        if not img_name.endswith((".png", ".jpg", ".jpeg")):
            img_name = img_name + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        label_name = row.get("id", row.get("label", row.get("identity")))
        label = self.label_map[label_name]

        img = load_image_with_optional_mask(
            img_path, self.use_alpha_mask, self.mask_dir
        )
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Test Dataset (for building embeddings)
# ---------------------------------------------------------------------------

class JaguarTestDataset(Dataset):
    """Test set: (image, image_name). Returns basename for embedding_dict keys."""

    def __init__(self, image_dir: str, transform=None,
                 use_alpha_mask: bool = False, mask_dir: str = None,
                 image_list: list = None):
        self.image_dir = image_dir.rstrip("/")
        self.transform = transform
        self.use_alpha_mask = use_alpha_mask
        self.mask_dir = mask_dir

        if image_list is not None:
            self.image_names = [os.path.basename(p) for p in image_list]
        else:
            exts = (".png", ".jpg", ".jpeg")
            self.image_names = sorted([
                f for f in os.listdir(self.image_dir)
                if f.lower().endswith(exts)
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, name)
        img = load_image_with_optional_mask(
            img_path, self.use_alpha_mask, self.mask_dir
        )
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, name


# ---------------------------------------------------------------------------
# Balanced sampling for class imbalance
# ---------------------------------------------------------------------------

def get_balanced_sampler(dataset: JaguarTrainDataset):
    """WeightedRandomSampler so each identity is seen roughly equally."""
    id_col = "id" if "id" in dataset.df.columns else "label" if "label" in dataset.df.columns else "individual_id"
    labels = [dataset.label_map[dataset.df.iloc[i][id_col]] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    weights = 1.0 / np.array(class_counts)[labels]
    return WeightedRandomSampler(weights, len(weights))
