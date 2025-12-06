import os
import sys
sys.path.append("..")
os.environ['TOKENIZERS_PARALLELISM'] = "False"
from typing import Dict, Union, List

import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_yml
from modeling_tara import TARA, read_frames_decord


def round_off(x):
    return str(np.round(x, 1))


def print_metrics_as_latex_row(metrics, v2t_metric='txt_r1', t2v_metric='img_map', sep=' & '):
    string = sep.join(
        [
            round_off(metrics['metrics_pair'][v2t_metric]),
            round_off(metrics['metrics_stat'][v2t_metric]),
            round_off(metrics['metrics_base'][v2t_metric]),
            round_off(metrics['metrics_pair'][t2v_metric]),
            round_off(metrics['metrics_stat'][t2v_metric]),
            round_off(metrics['metrics_base'][t2v_metric]),
        ],
    )
    print(string)


def load_data(dataset='ssv2'):
    
    # Load dataset config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    assert os.path.exists(cfg_path), f"Dataset config file {cfg_path} does not exist"
    data_config = load_yml(cfg_path)[dataset]
    
    csv_path = data_config['csv_path']
    video_dir = data_config['video_dir']
    ext = data_config['ext']

    # Load CSV
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    df = pd.read_csv(csv_path)

    # Add text ID
    df['text_id'] = df[['chiral_triplet_id', 'chiral_label']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1,
    )

    df['video_path'] = df['id'].apply(lambda x: f"{video_dir}/{x}.{ext}")
    df = df[df.video_path.apply(os.path.exists)]
    print("Number of rows: ", len(df))
    print("Sample row: ")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))

    return df


def _compute_mean_average_precision(scores: np.ndarray, query_to_relevants: Dict[int, Union[int, List[int]]]) -> float:
    """Compute mean average precision given a scores matrix and relevance mapping.

    Args:
        scores: 2D array shaped (num_queries, num_candidates)
        query_to_relevants: dict mapping each query index to either an int id
            or a list of relevant candidate ids.

    Returns:
        Mean Average Precision across queries that have at least one relevant.
        Returned as percentage (0-100).
    """
    num_queries, num_candidates = scores.shape
    assert len(query_to_relevants) == num_queries, \
        f"Mapping length {len(query_to_relevants)} != num_queries {num_queries}"

    average_precisions: List[float] = []

    for query_idx in range(num_queries):
        query_scores = scores[query_idx]
        assert query_scores.shape == (num_candidates,), \
            f"Expected shape ({num_candidates},), got {query_scores.shape}"

        sorted_candidate_indices = np.argsort(query_scores)[::-1]

        relevants = query_to_relevants[query_idx]
        if isinstance(relevants, int):
            relevant_set = {relevants}
        else:
            assert isinstance(relevants, (list, tuple)), \
                f"Relevant ids must be int or list/tuple, got {type(relevants)}"
            relevant_set = set(relevants)

        if len(relevant_set) == 0:
            # Skip queries with no relevant items
            continue

        num_retrieved_relevant = 0
        precision_sum = 0.0

        for rank, candidate_id in enumerate(sorted_candidate_indices, start=1):
            if candidate_id in relevant_set:
                num_retrieved_relevant += 1
                precision_at_k = num_retrieved_relevant / rank
                precision_sum += precision_at_k
                if num_retrieved_relevant == len(relevant_set):
                    # All relevants have been found; can stop early
                    break

        ap = precision_sum / len(relevant_set)
        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return 0.0

    return 100.0 * float(np.mean(average_precisions))


def itm_eval(
    scores_i2t: np.ndarray,
    scores_t2i: np.ndarray, 
    txt2img: Dict[int, Union[int, List[int]]],
    img2txt: Dict[int, Union[int, List[int]]],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate Image-Text Matching (ITM) performance for bidirectional retrieval tasks.
    
    This function computes standard retrieval metrics (Recall@1, Recall@5, Recall@10)
    for both image-to-text and text-to-image retrieval directions.
    
    Args:
        scores_i2t: Similarity scores for image-to-text retrieval.
                   Shape: (num_images, num_texts)
                   scores_i2t[i, j] = similarity between image i and text j
        scores_t2i: Similarity scores for text-to-image retrieval.
                   Shape: (num_texts, num_images) 
                   scores_t2i[i, j] = similarity between text i and image j
        txt2img: Mapping from text index to ground truth image ID(s).
                Keys are text indices, values are either:
                - int: single ground truth image ID
                - List[int]: multiple ground truth image IDs
        img2txt: Mapping from image index to ground truth text ID(s).
                Keys are image indices, values are either:
                - int: single ground truth text ID  
                - List[int]: multiple ground truth text IDs
        verbose: Whether to print verbose output.
    
    Returns:
        Dictionary containing evaluation metrics:
        - txt_r1, txt_r5, txt_r10: Image-to-text Recall@1,5,10 (%)
        - img_r1, img_r5, img_r10: Text-to-image Recall@1,5,10 (%)
        - txt_r_mean: Mean of image-to-text recalls
        - img_r_mean: Mean of text-to-image recalls  
        - r_mean: Overall mean of all recalls
        
    Note:
        - All metrics are returned as percentages (0-100) rounded to 2 decimal places
        - For multiple ground truths, the best (lowest) rank is used
        - Recall@K = % of queries where correct answer appears in top-K results
    """
    
    # Input validation
    assert isinstance(scores_i2t, np.ndarray), \
        f"scores_i2t must be numpy array, got {type(scores_i2t)}"
    assert isinstance(scores_t2i, np.ndarray), \
        f"scores_t2i must be numpy array, got {type(scores_t2i)}"
    assert scores_i2t.ndim == 2, \
        f"scores_i2t must be 2D, got shape {scores_i2t.shape}"
    assert scores_t2i.ndim == 2, \
        f"scores_t2i must be 2D, got shape {scores_t2i.shape}"
    
    num_images, num_texts = scores_i2t.shape
    num_texts_t2i, num_images_t2i = scores_t2i.shape
    
    assert num_texts == num_texts_t2i, \
        f"Text dimension mismatch: {num_texts} vs {num_texts_t2i}"
    assert num_images == num_images_t2i, \
        f"Image dimension mismatch: {num_images} vs {num_images_t2i}"
    assert len(img2txt) == num_images, \
        f"img2txt length {len(img2txt)} != num_images {num_images}"
    assert len(txt2img) == num_texts, \
        f"txt2img length {len(txt2img)} != num_texts {num_texts}"
    
    # ==========================================
    # Part 1: Image-to-Text Retrieval (I2T)
    # ==========================================
    if verbose:
        print(f"Computing image-to-text retrieval metrics for {num_images} images...")
    
    # Array to store rank of ground truth text for each image query
    i2t_ranks = np.zeros(num_images, dtype=np.float32)
    
    for img_idx in range(num_images):
        # Get similarity scores for this image against all texts
        img_scores = scores_i2t[img_idx]  # Shape: (num_texts,)
        assert img_scores.shape == (num_texts,), \
            f"Expected shape ({num_texts},), got {img_scores.shape}"
        
        # Sort text indices by similarity score in descending order
        # inds[0] = index of text with highest similarity to this image
        sorted_text_indices = np.argsort(img_scores)[::-1]  # Shape: (num_texts,)
        
        # Get ground truth text ID(s) for this image
        gt_text_ids = img2txt[img_idx]
        
        if isinstance(gt_text_ids, int):
            # Single ground truth text
            gt_text_ids = [gt_text_ids]
        else:
            # Multiple ground truth texts - ensure it's a list
            assert isinstance(gt_text_ids, (list, tuple)), \
                f"gt_text_ids must be int or list, got {type(gt_text_ids)}"
            gt_text_ids = list(gt_text_ids)
        
        # Find the best (lowest) rank among all ground truth texts
        best_rank = float('inf')
        for gt_text_id in gt_text_ids:
            # Find position of this ground truth text in the sorted ranking
            rank_positions = np.where(sorted_text_indices == gt_text_id)[0]
            assert len(rank_positions) == 1, \
                f"Text ID {gt_text_id} appears {len(rank_positions)} times in rankings"
            
            current_rank = rank_positions[0]  # 0-indexed rank (0 = best)
            if current_rank < best_rank:
                best_rank = current_rank
        
        i2t_ranks[img_idx] = best_rank
    
    # Compute image-to-text recall metrics
    # Recall@K = percentage of queries where GT appears in top-K results
    i2t_recall_at_1 = 100.0 * np.sum(i2t_ranks < 1) / len(i2t_ranks)
    i2t_recall_at_5 = 100.0 * np.sum(i2t_ranks < 5) / len(i2t_ranks) 
    i2t_recall_at_10 = 100.0 * np.sum(i2t_ranks < 10) / len(i2t_ranks)
    
    if verbose:
        print(f"I2T Results - R@1: {i2t_recall_at_1:.2f}%, R@5: {i2t_recall_at_5:.2f}%, R@10: {i2t_recall_at_10:.2f}%")
    
    # ==========================================
    # Part 2: Text-to-Image Retrieval (T2I)  
    # ==========================================
    if verbose:
        print(f"Computing text-to-image retrieval metrics for {num_texts} texts...")
    
    # Array to store rank of ground truth image for each text query
    t2i_ranks = np.zeros(num_texts, dtype=np.float32)
    
    for text_idx in range(num_texts):
        # Get similarity scores for this text against all images
        text_scores = scores_t2i[text_idx]  # Shape: (num_images,)
        assert text_scores.shape == (num_images,), \
            f"Expected shape ({num_images},), got {text_scores.shape}"
        
        # Sort image indices by similarity score in descending order
        # inds[0] = index of image with highest similarity to this text
        sorted_image_indices = np.argsort(text_scores)[::-1]  # Shape: (num_images,)
        
        # Get ground truth image ID(s) for this text
        gt_image_ids = txt2img[text_idx]
        
        if isinstance(gt_image_ids, int):
            # Single ground truth image
            gt_image_ids = [gt_image_ids]
        else:
            # Multiple ground truth images - ensure it's a list
            assert isinstance(gt_image_ids, (list, tuple)), \
                f"gt_image_ids must be int or list, got {type(gt_image_ids)}"
            gt_image_ids = list(gt_image_ids)
        
        # Find the best (lowest) rank among all ground truth images
        best_rank = float('inf')
        for gt_image_id in gt_image_ids:
            # Find position of this ground truth image in the sorted ranking
            rank_positions = np.where(sorted_image_indices == gt_image_id)[0]
            assert len(rank_positions) == 1, \
                f"Image ID {gt_image_id} appears {len(rank_positions)} times in rankings"
            
            current_rank = rank_positions[0]  # 0-indexed rank (0 = best)
            if current_rank < best_rank:
                best_rank = current_rank
        
        t2i_ranks[text_idx] = best_rank
    
    # Compute text-to-image recall metrics
    # Recall@K = percentage of queries where GT appears in top-K results
    t2i_recall_at_1 = 100.0 * np.sum(t2i_ranks < 1) / len(t2i_ranks)
    t2i_recall_at_5 = 100.0 * np.sum(t2i_ranks < 5) / len(t2i_ranks)
    t2i_recall_at_10 = 100.0 * np.sum(t2i_ranks < 10) / len(t2i_ranks)
    
    if verbose:
        print(f"T2I Results - R@1: {t2i_recall_at_1:.2f}%, R@5: {t2i_recall_at_5:.2f}%, R@10: {t2i_recall_at_10:.2f}%")
    
    # ==========================================
    # Part 3: Aggregate Final Metrics
    # ==========================================
    
    # Compute mean recalls for each direction
    i2t_mean_recall = (i2t_recall_at_1 + i2t_recall_at_5 + i2t_recall_at_10) / 3.0
    t2i_mean_recall = (t2i_recall_at_1 + t2i_recall_at_5 + t2i_recall_at_10) / 3.0
    
    # Compute overall mean across both directions
    overall_mean_recall = (i2t_mean_recall + t2i_mean_recall) / 2.0
    
    # Prepare results dictionary with descriptive names
    eval_results = {
        # Image-to-Text retrieval metrics
        "txt_r1": i2t_recall_at_1,      # Image-to-text Recall@1
        "txt_r5": i2t_recall_at_5,      # Image-to-text Recall@5  
        "txt_r10": i2t_recall_at_10,    # Image-to-text Recall@10
        "txt_r_mean": i2t_mean_recall,  # Mean of I2T recalls
        
        # Text-to-Image retrieval metrics
        "img_r1": t2i_recall_at_1,      # Text-to-image Recall@1
        "img_r5": t2i_recall_at_5,      # Text-to-image Recall@5
        "img_r10": t2i_recall_at_10,    # Text-to-image Recall@10  
        "img_r_mean": t2i_mean_recall,  # Mean of T2I recalls
        
        # Overall performance
        "r_mean": overall_mean_recall   # Overall mean across both directions
    }
    
    if verbose:
        print(f"Final Results - I2T Mean: {eval_results['txt_r_mean']:.2f}%, "
              f"T2I Mean: {eval_results['img_r_mean']:.2f}%, "
              f"Overall Mean: {eval_results['r_mean']:.2f}%")
    
    return eval_results


def itm_eval_with_map(
    scores_i2t: np.ndarray,
    scores_t2i: np.ndarray,
    txt2img: Dict[int, Union[int, List[int]]],
    img2txt: Dict[int, Union[int, List[int]]],
    verbose: bool = False,
) -> Dict[str, float]:
    """Extended ITM eval that also computes mAP for both directions.

    This function reuses `itm_eval` to compute recalls and then adds:
      - txt_map: Image-to-Text mAP (query=image, relevant texts)
      - img_map: Text-to-Image mAP (query=text, relevant images)
      - map_mean: Mean of txt_map and img_map
    All metrics are returned as percentages.
    """
    # Base recall metrics
    base_results = itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, verbose=verbose)

    # Compute mAP for both directions
    i2t_map = _compute_mean_average_precision(scores_i2t, img2txt)
    t2i_map = _compute_mean_average_precision(scores_t2i, txt2img)
    map_mean = (i2t_map + t2i_map) / 2.0

    extended = dict(base_results)
    extended.update({
        "txt_map": i2t_map,
        "img_map": t2i_map,
        "map_mean": map_mean,
    })

    if verbose:
        print(f"I2T mAP: {i2t_map:.2f}%, T2I mAP: {t2i_map:.2f}%, Mean mAP: {map_mean:.2f}%")

    return extended


def compute_retrieval_metrics_wrapper(
    df: pd.DataFrame, vid_feat: dict, text_feat: dict,
    text_id_col="text_id", video_id_col="id",
):
    # from chiral_retrieval.util.retrieval_general import (
    #     itm_eval_with_map,
    # )

    # Compute scores
    zv = torch.stack([vid_feat[k] for k in vid_feat]).numpy()
    zt = torch.stack([text_feat[k] for k in text_feat]).numpy()
    scores_v2t = zv @ zt.T
    scores_t2v = zt @ zv.T

    # Get matchings
    video_keys = np.array(list(vid_feat.keys()))
    text_keys = np.array(list(text_feat.keys()))

    # Text to video
    t2v = {}
    j = 0
    for k in text_feat:
        # For each sample, pick ONLY the videos that match with the caption
        matching_video_ids = df[(df[text_id_col] == k)][video_id_col].values
        t2v[j] = [np.where(x == video_keys)[0][0] for x in matching_video_ids]
        j += 1

    # Video to text
    v2t = {}
    j = 0
    for k in vid_feat:
        row = df[df[video_id_col] == k]
        if len(row) != 1:
            import ipdb; ipdb.set_trace()
        
        row = row.iloc[0].to_dict()
        text_id = row[text_id_col]
        v2t[j] = list(np.where(text_id == text_keys)[0])
        j += 1

    # return itm_eval(scores_v2t, scores_t2v, t2v, v2t, verbose=False)
    return itm_eval_with_map(scores_v2t, scores_t2v, t2v, v2t, verbose=False)


def weighted_average(df: pd.DataFrame, weight_col: str = "frac") -> pd.Series:
    """
    Compute the weighted average of all numeric columns in a DataFrame 
    using a specified weight column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns and a weight column.
    weight_col : str, default="frac"
        Column to use as weights. Must contain values between [0, 1].

    Returns
    -------
    pd.Series
        A Series containing the weighted averages for each numeric column 
        (excluding the weight column).
    """
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    weights = df[weight_col].to_numpy()
    if not np.isclose(weights.sum(), 1):
        weights = weights / weights.sum()  # normalize to sum=1 if needed

    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=[weight_col], errors="ignore")
    
    weighted_means = (numeric_cols * weights[:, None]).sum(axis=0)

    return weighted_means


def compute_retrieval_metrics_with_subsets(
    df: pd.DataFrame,
    vid_feat: dict,
    text_feat: dict,
    agg_col: str,
    video_id_col: str = 'id',
    text_id_col: str = 'text_id',
    verbose: bool = False,
):
    """
    Computes retrieval with different subsets of df and then
    aggregates the results.
    
    Args:
        df: pd.DataFrame, CSV with all the information.
        vid_feat: dict, Video features.
        text_feat: dict, Text features.
        agg_col: str, Column name for aggregation.
        verbose: bool, Whether to print progress.

    Returns:
        pd.DataFrame, Aggregated retrieval results.
    """
    assert agg_col in df.columns
    assert video_id_col in df.columns
    assert text_id_col in df.columns
    
    values = df[agg_col].unique()
    results = []
    for v in values:
        
        # Collect video and text features only for this value of agg_col
        _video_ids = df[df[agg_col] == v][video_id_col].values
        _text_ids = df[df[agg_col] == v][text_id_col].values

        _vid_feat = {k: vid_feat[k] for k in _video_ids}
        _text_feat = {k: text_feat[k] for k in _text_ids}
        if verbose:
            print(f"Computing retrieval metrics for {agg_col} = {v}")
            print(f"Number of video IDs: {len(np.unique(_video_ids))}")
            print(f"Number of text IDs: {len(np.unique(_text_ids))}")
            print('-' * 100)
        
        # Compute retrieval metrics
        _results = compute_retrieval_metrics_wrapper(
            df[df[agg_col] == v],
            _vid_feat,
            _text_feat,
            video_id_col=video_id_col,
            text_id_col=text_id_col,
        )
        _results[agg_col] = v
        _results['frac'] = len(_vid_feat) / len(df)
        results.append(_results)
    results = pd.DataFrame(results)
    
    # Average the results: weighted by the fraction of samples in each subset
    results_avg = weighted_average(results, weight_col='frac').to_dict()
    # results_avg = results.mean().to_dict()

    return results_avg

def compute_metrics(df, _video_feats, _texts_feats, show_metrics=True):
    df['base'] = 'base'
    metrics_base = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='base', verbose=False,
    )
    metrics_pair = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='chiral_triplet_id', verbose=False,
    )
    metrics_stat = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='chiral_label', verbose=False,
    )
    if show_metrics:
        print(json.dumps(metrics_pair, indent=4))
        print(json.dumps(metrics_stat, indent=4))
        print(json.dumps(metrics_base, indent=4))
    return dict(
        metrics_pair=metrics_pair,
        metrics_stat=metrics_stat,
        metrics_base=metrics_base,
    )


if __name__ == "__main__":
    
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--dataset', type=str, default='ssv2')
    parser.add_argument('--eval_on_subset', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load data
    debug = args.debug
    eval_on_subset = args.eval_on_subset
    df = load_data(dataset=args.dataset)
    df = df.drop_duplicates(subset=['id', 'text_id']).reset_index(drop=True)
    if eval_on_subset:
        df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        print(f"Evaluating on {len(df)} samples only.")
    else:
        print(f"Evaluating on all {len(df)} samples.")
    if debug:
        row = df.iloc[0]
    
    # Load model
    model = TARA.from_pretrained(
        args.model_path,  # Load from current directory
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    # Compute text features
    text_ids = df['text_id'].unique()
    texts_feat = {}
    for text_id in tqdm_iterator(text_ids, desc='Computing text features'):
        text = df[df.text_id == text_id].template.unique()[0]
        with torch.no_grad():
            zt = model.encode_text(text).cpu().float().squeeze(0)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_feat[text_id] = zt

    # Compute video features
    video_paths = df.video_path.unique()
    video_ids = df.id.unique()
    video_feat = {}
    j = 0
    for video_path in tqdm_iterator(video_paths, desc='Computing video features'):
        with torch.no_grad():
            zv = model.encode_vision(
                read_frames_decord(video_path, num_frames=16).unsqueeze(0)
            ).cpu().float().squeeze(0)
        zv = torch.nn.functional.normalize(zv, dim=-1)
        video_feat[video_ids[j]] = zv
        j += 1

    metrics = compute_metrics(df, video_feat, texts_feat, show_metrics=False)
    print(json.dumps(metrics, indent=4))
    print('-' * 100)
    print_metrics_as_latex_row(metrics, sep='& ')
    print('-' * 100)
