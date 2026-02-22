"""
Inference: build embedding dict from test images and create submission.csv.
"""
import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JaguarTestDataset, get_test_transform
from models import ReIDModel
from train import get_config, build_embedding_dict


def run_inference(
    checkpoint_path,
    test_dir,
    test_csv_path,
    submission_path="/kaggle/working/submission.csv",
    device=None,
    batch_size=32,
    num_workers=2,
    img_size=224,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    embedding_dim = ckpt["embedding_dim"]

    model = ReIDModel(embedding_dim=embedding_dim, backbone="resnet50", pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    test_dataset = JaguarTestDataset(test_dir, transform=get_test_transform(img_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    embedding_dict = build_embedding_dict(model, test_loader, device, normalize=True)

    test_csv = pd.read_csv(test_csv_path)
    similarities = []
    for _, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Similarity"):
        q_name = row["query_image"]
        g_name = row["gallery_image"]
        # Ensure we use same key format as in dataset (basename)
        q_key = os.path.basename(q_name)
        g_key = os.path.basename(g_name)
        q_emb = embedding_dict[q_key]
        g_emb = embedding_dict[g_key]
        sim = F.cosine_similarity(q_emb.unsqueeze(0), g_emb.unsqueeze(0)).item()
        similarities.append(sim)

    submission = pd.DataFrame({
        "row_id": test_csv["row_id"],
        "similarity": similarities,
    })
    submission.to_csv(submission_path, index=False)
    print(f"Saved to {submission_path}")
    return submission, embedding_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/best_model.pt")
    parser.add_argument("--test-dir", type=str, default="/kaggle/input/jaguar-re-id/test/test")
    parser.add_argument("--test-csv", type=str, default="/kaggle/input/jaguar-re-id/test.csv")
    parser.add_argument("--output", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = get_config(kaggle=True)
    run_inference(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        test_csv_path=args.test_csv,
        submission_path=args.output,
        batch_size=args.batch_size,
        num_workers=cfg["num_workers"],
        img_size=cfg["img_size"],
    )


if __name__ == "__main__":
    main()
