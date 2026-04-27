import os
import sys

import torch
import pandas as pd
import numpy as np
import json
from collections import defaultdict

import shared.utils as su
from tabulate import tabulate


def get_distractor_ids(subdf, q, mode='chiral'):
    assert mode in ['chiral', 'static', 'all']

    if mode == "chiral":
        a, b = q.split("_")
        if b == '0.0':
            b = '1.0'
        else:
            b = '0.0'
        q_ = f"{a}_{b}"
        z = [q_]

    elif mode == "static":
        a, b = q.split("_")
        _all = subdf[subdf.modality == 'text'].id.unique()
        z = [x for x in _all if (x.split("_")[1] == b and x != q)]

    elif mode == "all":
        z = list(set(subdf[subdf.modality == 'text'].id.unique()) - {q})

    else:
        raise ValueError

    return np.array(z)


def compute_retrieval_metrics(sim: torch.Tensor, labels: torch.Tensor):
    """
    Compute R@1, R@5, R@10 and mAP for a single query.

    R@K:  1 if *any* relevant candidate is in the top-K, else 0.
    mAP:  Average Precision over the full ranked list.

    Args:
        sim:    (C,) similarity scores.
        labels: (C,) binary relevance labels.

    Returns:
        dict with R@1, R@5, R@10 (int: 0 or 1), and mAP (float).
    """
    sorted_labels = labels[sim.argsort(descending=True)].float()
    C = sorted_labels.shape[0]

    # R@K — 1 if any relevant item exists in top-K
    cumsum = sorted_labels.cumsum(0)
    ks = [1, 5, 10]
    hits = cumsum[torch.tensor([min(k, C) - 1 for k in ks], device=sim.device)]

    # AP — precision at each relevant position, averaged
    num_relevant = cumsum[-1]
    ranks = torch.arange(1, C + 1, device=sim.device, dtype=sim.dtype)
    precision_at_hits = cumsum / ranks
    ap = (precision_at_hits * sorted_labels).sum() / num_relevant.clamp(min=1)

    return {
        "R@1":  int(hits[0] > 0),
        "R@5":  int(hits[1] > 0),
        "R@10": int(hits[2] > 0),
        "mAP":  ap.item(),
    }


def compute_metrics_time_t2v(df, feat, labels, dataset, feat_b=None, alpha=0.5):
    """
    Computes text-to-video retrieval metrics for the time nuance.

    If feat_b is given, similarity is alpha * sim(feat) + (1 - alpha) * sim(feat_b).
    """

    subdf = df[(df['nuance'] == 'time') & (df.source == f'cia-{dataset}')]

    # {"chiral": {"R@1": [...], ...}}
    metrics = defaultdict(lambda: defaultdict(list))

    modes = ['chiral', 'static', 'all']
    for mode in modes:
        for q in su.log.tqdm_iterator(subdf[subdf.modality == 'text'].id.unique(), desc=f"Computing metrics for {mode}: "):
            distractor_ids = get_distractor_ids(subdf, q, mode)

            # Get query embedding
            zq = feat[q]

            # First, get all the candidate IDs & corr. binary labels (whether or not it is a match)
            candidate_ids = [*labels[q]]
            is_match = [*([1] * len(labels[q]))]
            for q_ in distractor_ids:
                candidate_ids.extend(labels[q_])
                is_match.extend([0] * len(labels[q_]))

            # Get candidate embeddings
            zc = torch.stack([feat[str(x)] for x in candidate_ids])

            # Compute similarity
            s = zq @ zc.T
            if feat_b is not None:
                zq_b = feat_b[q]
                zc_b = torch.stack([feat_b[str(x)] for x in candidate_ids])
                s = alpha * s + (1 - alpha) * (zq_b @ zc_b.T)

            # Compute metrics
            m = compute_retrieval_metrics(s, torch.tensor(is_match))

            for k, v in m.items():
                metrics[mode][k].append(v)

    avg = {
        k: {m: round(sum(v) / len(v) * 100, 2) for m, v in inner.items()}
        for k, inner in metrics.items()
    }
    return avg


def compute_metrics_time_v2t(df, feat, labels, dataset, feat_b=None, alpha=0.5):
    subdf = df[(df['nuance'] == 'time') & (df.source == f'cia-{dataset}')]
    metrics = defaultdict(lambda: defaultdict(list))
    modes = ['chiral', 'static', 'all']

    for mode in modes:
        for q in su.log.tqdm_iterator(subdf[subdf.modality == 'video'].id.unique(), desc=f"Computing metrics for {mode}: "):

            # Only a single matching ID
            match_id = labels[q][0]
            dist_ids = get_distractor_ids(subdf, match_id, mode)
            candidate_ids = [match_id] + list(dist_ids)
            is_match = [1] + [0] * len(dist_ids)
            
            zq = feat[q]

            # Get candidate embeddings
            zc = torch.stack([feat[str(x)] for x in candidate_ids])

            # Compute similarity
            s = zq @ zc.T
            if feat_b is not None:
                zq_b = feat_b[q]
                zc_b = torch.stack([feat_b[str(x)] for x in candidate_ids])
                s = alpha * s + (1 - alpha) * (zq_b @ zc_b.T)

            # Compute metrics
            m = compute_retrieval_metrics(s, torch.tensor(is_match))

            for k, v in m.items():
                metrics[mode][k].append(v)

    avg = {
        k: {m: round(sum(v) / len(v) * 100, 2) for m, v in inner.items()}
        for k, inner in metrics.items()
    }
    return avg


def _negation_nan_block():
    """Placeholder when a negation split has no rows in the dataframe (JSON-safe)."""
    return {"R@1": None, "R@5": None, "R@10": None, "mAP": None}


def _empty_negation_mode_metrics(mode: str):
    return {mode: _negation_nan_block()}


def compute_metrics_negation(df, feat, labels, dataset, feat_b=None, alpha=0.5):
    subdf = df[(df.nuance == 'negation') & (df.source == f'neg-{dataset}')]
    modality = "image" if dataset == "coco" else "video"

    modes = ['standard', 'negation']
    metrics_all = {}

    for mode in modes:
        query_ids = subdf[subdf.modality == f"text-{mode}"].id.unique()
        candidate_ids = subdf[subdf.modality == modality].id.unique()
        if len(candidate_ids) == 0:
            metrics_all[mode] = _empty_negation_mode_metrics(mode)
            continue

        zc = torch.stack([feat[x] for x in candidate_ids])
        zc_b = torch.stack([feat_b[x] for x in candidate_ids]) if feat_b is not None else None

        metrics = defaultdict(lambda: defaultdict(list))
        for q in su.log.tqdm_iterator(query_ids, desc=f"Computing metrics for {mode}"):
            if q not in feat or (feat_b is not None and q not in feat_b):
                continue
            zq = feat[q]
            s = zq @ zc.T
            if feat_b is not None:
                zq_b = feat_b[q]
                s = alpha * s + (1 - alpha) * (zq_b @ zc_b.T)
            lab = labels[q]
            is_match = np.array([x in lab for x in candidate_ids]).astype(int)
            
            # Compute metrics
            m = compute_retrieval_metrics(s, torch.tensor(is_match))
        
            for k, v in m.items():
                metrics[mode][k].append(v)
        
        avg = {
            k: {m: round(sum(v) / len(v) * 100, 2) for m, v in inner.items()}
            for k, inner in metrics.items()
        }
        if not avg:
            metrics_all[mode] = _empty_negation_mode_metrics(mode)
        else:
            metrics_all[mode] = avg
    return metrics_all


def compute_metrics_multimodal_covr(df, feat, labels, feat_b=None, alpha=0.5):
    subdf = df[(df.nuance == 'multimodal') & (df.source == 'covr-webvid')]
    query_ids = subdf[subdf.modality == "video-text"].id.unique()
    candidate_ids = subdf[subdf.modality == "video"].id.unique()
    S = torch.stack([feat[x] for x in query_ids]) @ torch.stack([feat[x] for x in candidate_ids]).T
    if feat_b is not None:
        S_b = torch.stack([feat_b[x] for x in query_ids]) @ torch.stack(
            [feat_b[x] for x in candidate_ids]
        ).T
        S = alpha * S + (1 - alpha) * S_b
    metrics = defaultdict(lambda: defaultdict(list))
    mode = "covr"
    for i in su.log.tqdm_iterator(range(len(query_ids)), desc="Computing metrics for WebVid-CoVR"):
        s = S[i]
        q = query_ids[i]
        is_match = np.array([x in labels[q] for x in candidate_ids]).astype(int)
        # Compute metrics
        m = compute_retrieval_metrics(s, torch.tensor(is_match))

        for k, v in m.items():
            metrics[mode][k].append(v)
    avg = {
        k: {m: round(sum(v) / len(v) * 100, 2) for m, v in inner.items()}
        for k, inner in metrics.items()
    }
    return avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/')
    parser.add_argument('--model_name', type=str, default='tarsier2_7b')
    parser.add_argument('--feat_path', type=str, default=None)
    parser.add_argument('--csv_path', type=str, default='./data/nuanced_retrieval_inputs-val.csv')
    parser.add_argument('--lab_path', type=str, default='./data/nuanced_retrieval_labels-val.json')
    args = parser.parse_args()
    
    if args.feat_path is None:
        assert args.model_path is not None, "model_path is required if feat_path is not provided"
        assert args.model_name is not None, "model_name is required if feat_path is not provided"
    else:
        args.model_path = None
        args.model_name = None

    csv_path = args.csv_path
    csv_name = os.path.basename(csv_path).split('.')[0]
    lab_path = args.lab_path
    csv_name = os.path.basename(csv_path).split('.')[0]

    assert os.path.exists(csv_path), f"CSV file does not exist: {csv_path}"
    df = pd.read_csv(csv_path)
    labels = su.io.load_json(lab_path)
    print(f"Loaded {len(df)} rows from {csv_path}.")
    print(f"Loaded {len(labels)} labels.")
    
    # Load features
    if args.feat_path is None:
        path = f"{args.model_path}/embs/{args.model_name}_{csv_name}_embeddings.pt"
    else:
        path = args.feat_path
    assert os.path.exists(path), f"Features file does not exist: {path}"
    feat = torch.load(path)
    print(f"Loaded {len(feat)} features.")
    
    # For validation, for temporal nuance, we only have SSv2.
    if csv_name == 'nuanced_retrieval_inputs-val':
        time_nuance_datasets = ['ssv2']
        negn_nuance_datasets = ['msrvtt']
    else:
        time_nuance_datasets = ['ssv2', 'epic', 'charades']
        negn_nuance_datasets = ['coco', 'msrvtt']
    
    
    # Compute metrics one by one
    metrics = {}
    
    # Time
    for dataset in time_nuance_datasets:
        metrics[f'time_t2v-{dataset}'] = compute_metrics_time_t2v(df, feat, labels, dataset)
        metrics[f'time_v2t-{dataset}'] = compute_metrics_time_v2t(df, feat, labels, dataset)
    
    # Negation
    for dataset in negn_nuance_datasets:
        metrics[f'negation-{dataset}'] = compute_metrics_negation(df, feat, labels, dataset)
    
    # Multimodal
    metrics['multimodal_covr'] = compute_metrics_multimodal_covr(df, feat, labels)
    
    # Save metrics
    if args.feat_path is None:
        result_dir = f"{args.model_path}/metrics"
    else:
        result_dir = os.path.join(os.path.dirname(os.path.dirname(args.feat_path)), "metrics")
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f"metrics_{args.model_name}_{csv_name}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {save_path}")
    
    # Print all the key metrics
    keys = [f'time_v2t-{d}/chiral/R@1' for d in time_nuance_datasets]
    for d in negn_nuance_datasets:
        keys.append(f'negation-{d}/standard/standard/R@5')
        keys.append(f'negation-{d}/negation/negation/R@5')
    keys.append('multimodal_covr/covr/R@1')
    vals = []
    for k in keys:
        try:
            if k.count('/') == 2:
                m = metrics[k.split('/')[0]][k.split('/')[1]][k.split('/')[2]]
            elif k.count('/') == 3:
                m = metrics[k.split('/')[0]][k.split('/')[1]][k.split('/')[2]][k.split('/')[3]]
            else:
                raise ValueError(f"Invalid key: {k}")
        except Exception as e:
            print(f"Error getting metric for {k}: {e}")
        vals.append(m)
    vals = [str(v) for v in vals]
    rows = [
        [k, v] for k, v in zip(keys, vals)
    ]
    table = tabulate(rows, tablefmt="grid")
    print(table)