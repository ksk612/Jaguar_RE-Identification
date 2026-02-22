# Jaguar Re-ID – Improved Baseline

Improvements over a basic ArcFace Re-ID setup for the **Jaguar Identification** Kaggle competition (identity-balanced mAP, retrieval).

---

## Improvements at a glance

| Area | Change | Why |
|------|--------|-----|
| **Data** | ReID-style augmentations | ColorJitter, RandomGrayscale, RandomErasing reduce reliance on background and lighting; encourage learning spot patterns. |
| **Data** | Optional alpha mask | Composite jaguar onto neutral background to reduce spurious correlations (riverbank, vegetation). |
| **Data** | Identity-balanced sampling | `WeightedRandomSampler` so rare jaguars are seen as often as frequent ones (handles data imbalance). |
| **Model** | Backbone returns only embedding | `ReIDModel` returns a single tensor (no tuple); clean interface with ArcFace head and inference. |
| **Training** | AdamW + cosine LR | Better convergence and generalization. |
| **Training** | Gradient clipping | Stabilizes training with ArcFace. |
| **Inference** | Correct `embedding_dict` | Built from test loader with **basename** keys so `query_image` / `gallery_image` match; L2-normalized once. |
| **Eval** | Identity-balanced mAP | Metric in `metrics.py` for validation when identity labels are available. |

---

## Project layout

```
jaguar-reid/
├── dataset.py    # JaguarTrainDataset, JaguarTestDataset, transforms, balanced sampler, optional mask
├── models.py     # ReIDModel (backbone), ArcFace head
├── metrics.py    # Identity-balanced mAP
├── train.py      # Training loop, checkpointing
├── inference.py  # Build embedding dict, write submission.csv
├── requirements.txt
└── README.md
```

---

## Kaggle setup

1. **Dataset**: Add competition data (e.g. **Jaguar Re-ID**).
2. **Paths** in `get_config(kaggle=True)` in `train.py`:
   - `train_csv`, `train_dir`, `test_dir`, `test_csv` (adjust if your dataset uses different names).
3. **CSV columns**: Train CSV should have an identity column: `id`, `label`, or `individual_id`, and an image column: `image`, `image_name`, or `file`. Test CSV should have `query_image`, `gallery_image`, `row_id`.

---

## Run on Kaggle

**Training (notebook or script):**

```python
# In a Kaggle notebook, run:
!python train.py --epochs 20 --batch-size 32 --lr 1e-4 --backbone resnet50
# Or import and call main() / run cells equivalent to train.py
```

**Inference & submission:**

```python
!python inference.py --checkpoint /kaggle/working/best_model.pt \
  --test-dir /kaggle/input/jaguar-re-id/test/test \
  --test-csv /kaggle/input/jaguar-re-id/test.csv \
  --output /kaggle/working/submission.csv
```

Or in notebook form:

```python
from inference import run_inference

submission, embedding_dict = run_inference(
    checkpoint_path="/kaggle/working/best_model.pt",
    test_dir="/kaggle/input/jaguar-re-id/test/test",
    test_csv_path="/kaggle/input/jaguar-re-id/test.csv",
    submission_path="/kaggle/working/submission.csv",
)
```

---

## Optional: use alpha mask

If the competition provides masks (e.g. alpha channel or `*_mask.png`):

- Set `use_alpha_mask=True` and `mask_dir` in config.
- In `dataset.py`, `load_image_with_optional_mask` composites the jaguar onto a gray background so the model focuses on the animal.

---

## Optional: validation mAP

If you have a validation set with identity labels (e.g. `query_id`, `gallery_id`), you can compute identity-balanced mAP:

```python
from metrics import retrieval_map_from_embeddings

mAP, sims = retrieval_map_from_embeddings(
    embedding_dict, val_df,
    query_col="query_image", gallery_col="gallery_image",
    query_id_col="query_id", gallery_id_col="gallery_id",
)
print("Identity-balanced mAP:", mAP)
```

---

## Suggested next steps

- Try **MegaDescriptor-Large** (competition baseline) as backbone if available.
- Tune **ArcFace** `s` and `m` (e.g. `s=32`, `m=0.45`).
- Add **validation split** and early stopping by mAP.
- **Larger resolution** (e.g. 384) if GPU memory allows.
- **Test-time augmentation** (e.g. multi-crop or flip and average embeddings).
