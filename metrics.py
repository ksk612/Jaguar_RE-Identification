"""
Identity-balanced mAP for Re-ID evaluation (class-imbalance aware).
"""
import numpy as np


def identity_balanced_ap(y_true, y_score, identities):
    """
    Average Precision for one identity. Then we average over identities.
    y_true: binary (1 = same identity as query)
    y_score: similarity scores (higher = more similar)
    identities: identity id per sample (same length as y_true/y_score)
    """
    order = np.argsort(-np.asarray(y_score))
    y_true = np.asarray(y_true)[order]
    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0
    prec = np.cumsum(y_true) / np.arange(1, len(y_true) + 1, dtype=np.float64)
    ap = np.sum(prec * y_true) / n_pos
    return ap


def identity_balanced_map(queries, galleries, query_ids, gallery_ids, similarities):
    """
    Compute mAP where AP is computed per query identity and then averaged.

    queries: list of query indices (or names) of length N
    galleries: list of gallery indices (or names) of length N (one per query)
    query_ids: identity per query (array/list length N)
    gallery_ids: identity per gallery (array/list, full gallery set - need mapping from gallery index to id)
    similarities: N-length array of similarity between query and its gallery for this row

    For the competition format we often have a table with query_image, gallery_image, and we need
    to compute mAP over all (query, gallery) pairs. Simpler form: for each query we have one
    gallery (the pair). So we need to know for each row: is query_id == gallery_id? That gives
    y_true. y_score is the similarity. Then AP per identity: group by query identity, compute AP
    per group, then average.

    Simplified API for a DataFrame with columns: query_id, gallery_id, similarity
    """
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    similarities = np.asarray(similarities)
    y_true = (query_ids == gallery_ids).astype(np.float64)

    unique_ids = np.unique(query_ids)
    aps = []
    for uid in unique_ids:
        mask = query_ids == uid
        if mask.sum() == 0:
            continue
        aps.append(identity_balanced_ap(y_true[mask], similarities[mask], gallery_ids[mask]))
    return np.mean(aps) if aps else 0.0


def retrieval_map_from_embeddings(embedding_dict, test_df, query_col="query_image", gallery_col="gallery_image",
                                  query_id_col="query_id", gallery_id_col="gallery_id", similarity_fn=None):
    """
    test_df has query_image, gallery_image, and identity columns (query_id, gallery_id) for validation.
    similarity_fn(emb_q, emb_g) -> scalar. Default: cosine.
    Returns: (identity_balanced_mAP, list of similarities).
    """
    import torch
    import torch.nn.functional as F

    if similarity_fn is None:
        def similarity_fn(q, g):
            if isinstance(q, torch.Tensor):
                q = q.unsqueeze(0) if q.dim() == 1 else q
            if isinstance(g, torch.Tensor):
                g = g.unsqueeze(0) if g.dim() == 1 else g
            return F.cosine_similarity(q, g).item()

    sims = []
    for _, row in test_df.iterrows():
        q = embedding_dict[row[query_col]]
        g = embedding_dict[row[gallery_col]]
        sim = similarity_fn(q, g)
        sims.append(sim)

    if query_id_col not in test_df.columns or gallery_id_col not in test_df.columns:
        return np.mean(sims), sims

    return identity_balanced_map(
        test_df[query_col].values,
        test_df[gallery_col].values,
        test_df[query_id_col].values,
        test_df[gallery_id_col].values,
        sims
    ), sims
