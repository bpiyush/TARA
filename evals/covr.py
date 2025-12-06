"""Composed Video Retrieval (CoVR) Evaluation"""
import os
import sys
import json
from typing import Dict

sys.path.append("..")
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import decord

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_yml
from modeling_tara import TARA, read_frames_decord


def load_data(dataset: str = 'covr') -> pd.DataFrame:
    """
    Load dataset configuration and CSV file for CoVR.
    
    Args:
        dataset: Name of dataset to load (covr)
        
    Returns:
        DataFrame with video paths and metadata
    """
    # Load dataset config from YAML
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    assert os.path.exists(cfg_path), f"Dataset config file {cfg_path} does not exist"
    all_configs = load_yml(cfg_path)
    
    # Validate dataset
    assert dataset in all_configs, f"Dataset {dataset} not found in datasets.yaml"
    
    data_config = all_configs[dataset]
    
    # Load CSV
    csv_path = data_config['csv_path']
    video_dir = data_config['video_dir']
    
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {dataset}")
    print(f"Number of rows in CoVR-test: {len(df)}")
    
    # Add video paths
    df['video1_path'] = df['video1'].apply(lambda x: f"{video_dir}/{x}")
    df['video2_path'] = df['video2'].apply(lambda x: f"{video_dir}/{x}")
    
    # Filter to only existing videos
    df = df[df.video1_path.apply(os.path.exists) & df.video2_path.apply(os.path.exists)]
    print(f"Number of rows with all videos available: {df.shape}")
    
    # Remove problematic videos (if any)
    problematic_videos = data_config.get('problematic_videos', [])
    for video_path in problematic_videos:
        df = df[df.video1_path != video_path]
    
    print(f"Sample row:")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))
    
    return df


def embed_composed_query(
    model: TARA,
    video_path: str,
    edit_text: str,
    verbose: bool = False
) -> torch.Tensor:
    """
    Embed a composed query using video and edit text.
    
    Args:
        model: TARA model instance
        video_path: Path to source video
        edit_text: Edit instruction text
        verbose: Whether to print debug info
        
    Returns:
        Query embedding tensor
    """
    # Read middle frame from video
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path, num_threads=1)
    mid = len(vr) // 2
    image = torch.from_numpy(vr[mid].asnumpy()).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # [T, C, H, W]
    
    if verbose:
        print(f"Using edit text: {edit_text}")
        print("-" * 120)
    
    # Encode using TARA's encode_vision with edit_text parameter
    with torch.no_grad():
        zv = model.encode_vision(image, edit_text=edit_text).squeeze(0)
    
    if verbose:
        print(f"Query embedding shape: {zv.shape}")
    
    return zv


def gather_metrics(
    query_embeds: Dict[str, torch.Tensor],
    candidates: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute retrieval metrics for CoVR.
    
    Args:
        query_embeds: Dictionary of query embeddings
        candidates: Dictionary of candidate video embeddings
        df: DataFrame with ground truth mappings
        verbose: Whether to print progress
        
    Returns:
        Dictionary of retrieval metrics
    """
    from cia import itm_eval
    
    zq = []
    zc = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        query_key = f"{row['edit']}|{row['video1']}"
        candi_key = row['video2']
        if query_key not in query_embeds or candi_key not in candidates:
            if verbose:
                print(f"Missing value for {i}. Skipped.")
            continue
        zq.append(query_embeds[query_key])
        zc.append(candidates[candi_key])
    
    zq = torch.stack(zq).numpy()
    zc = torch.stack(zc).numpy()
    
    if verbose:
        print(f"Query embeddings shape: {zq.shape}")
        print(f"Candidate embeddings shape: {zc.shape}")
    
    # i:q and t:c; and we care about q2c metrics, i.e., i2t, i.e., text_*
    score_q2c = zq @ zc.T
    score_c2q = zc @ zq.T
    indices = {i: i for i in range(len(score_q2c))}
    metrics = itm_eval(
        scores_i2t=score_q2c,
        scores_t2i=score_c2q,
        txt2img=indices,
        img2txt=indices,
        verbose=verbose
    )
    return metrics


if __name__ == "__main__":
    
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to TARA model checkpoint')
    parser.add_argument('--device_map', type=str, default='auto',
                       help='Device map for model loading')
    parser.add_argument('--dataset', type=str, default='covr',
                       choices=['covr'],
                       help='Dataset to evaluate on')
    parser.add_argument('--num_frames', type=int, default=15,
                       help='Number of frames to sample from candidate videos')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - only process 100 samples')
    args = parser.parse_args()

    # Load data
    df = load_data(dataset=args.dataset)
    
    # Load dataset config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    all_configs = load_yml(cfg_path)
    data_config = all_configs[args.dataset]
    video_dir = data_config['video_dir']
    
    # Debug mode
    if args.debug:
        df = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: Evaluating on {len(df)} samples only.")
    else:
        print(f"Evaluating on all {len(df)} samples.")
    
    print('-' * 100)
    
    # Load model
    print(f"Loading TARA model from {args.model_path}...")
    model = TARA.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    print("Model loaded successfully.")
    print('-' * 100)

    # Compute query embeddings
    print(f"Computing query embeddings for {len(df)} queries...")
    query_embeddings = {}
    failed_queries = []
    
    for i in tqdm_iterator(range(len(df)), desc='Computing query embeddings'):
        row = df.iloc[i].to_dict()
        video_path = row['video1_path']
        edit_text = row['edit']
        
        try:
            with torch.no_grad():
                zv = embed_composed_query(model, video_path, edit_text)
                zv = torch.nn.functional.normalize(zv, dim=-1)
                zv = zv.cpu().float()
            
            key = f"{edit_text}|{row['video1']}"
            query_embeddings[key] = zv
        except Exception as e:
            print(f"Failed to process query {i}: {str(e)}")
            failed_queries.append(i)
            continue
    
    print(f"Successfully computed {len(query_embeddings)} query embeddings.")
    if len(failed_queries) > 0:
        print(f"Failed to process {len(failed_queries)} queries.")
    print('-' * 100)

    # Compute candidate video embeddings
    videos = set(df.video2.tolist())
    print(f"Computing embeddings for {len(videos)} candidate videos...")
    candidates = {}
    failed_videos = []
    
    for video in tqdm_iterator(videos, desc='Computing candidate video embeddings'):
        video_path = f"{video_dir}/{video}"
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            failed_videos.append(video)
            continue
        
        try:
            with torch.no_grad():
                zv = model.encode_vision(
                    read_frames_decord(video_path, num_frames=args.num_frames).unsqueeze(0)
                ).cpu().float().squeeze(0)
                zv = torch.nn.functional.normalize(zv, dim=-1)
            candidates[video] = zv
        except Exception as e:
            print(f"Failed to process video {video}: {str(e)}")
            failed_videos.append(video)
            continue
    
    print(f"Successfully computed {len(candidates)} candidate embeddings.")
    if len(failed_videos) > 0:
        print(f"Failed to process {len(failed_videos)} videos.")
    print('-' * 100)

    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    metrics = gather_metrics(query_embeddings, candidates, df, verbose=True)
    
    print('-' * 100)
    print("Final Results:")
    print(json.dumps(metrics, indent=4))
    print('-' * 100)
    
    # Save results to file
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/covr_{args.dataset}.json"
    
    # Add metadata to results
    metrics['dataset'] = args.dataset
    metrics['model_path'] = args.model_path
    metrics['num_frames'] = args.num_frames
    metrics['debug'] = args.debug
    
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {result_path}")
