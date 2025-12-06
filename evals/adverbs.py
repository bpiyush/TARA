"""Adverb Recognition Evaluation"""
import os
import sys
import json
from typing import Dict
from glob import glob

sys.path.append("..")
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from natsort import natsorted
plt.rcParams["font.family"] = "serif"

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_yml
from modeling_tara import TARA


def load_data(dataset: str = 'vatex-adverbs') -> tuple:
    """
    Load dataset configuration and CSV files for adverb recognition.
    
    Args:
        dataset: Name of dataset to load (vatex-adverbs)
        
    Returns:
        Tuple of (annotations_df, adverbs_df, data_config)
    """
    # Load dataset config from YAML
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    assert os.path.exists(cfg_path), f"Dataset config file {cfg_path} does not exist"
    all_configs = load_yml(cfg_path)
    
    # Validate dataset
    assert dataset in all_configs, f"Dataset {dataset} not found in datasets.yaml"
    
    data_config = all_configs[dataset]
    
    # Load CSVs
    annotations_csv = data_config['annotations_csv']
    adverbs_csv = data_config['adverbs_csv']
    frame_dir = data_config['frame_dir']
    
    assert os.path.exists(annotations_csv), f"Annotations CSV {annotations_csv} does not exist"
    assert os.path.exists(adverbs_csv), f"Adverbs CSV {adverbs_csv} does not exist"
    
    df_anno = pd.read_csv(annotations_csv)
    df_advb = pd.read_csv(adverbs_csv)
    
    print(f"Dataset: {dataset}")
    print(f"Annotations shape: {df_anno.shape}")
    print(f"Adverbs shape: {df_advb.shape}")
    
    # Add clip directory path
    clip_id_col = data_config['clip_id_col']
    df_anno['clip_dir'] = df_anno[clip_id_col].apply(lambda x: f"{frame_dir}/{x}")
    
    # Filter to only existing clip directories
    df_anno = df_anno[df_anno['clip_dir'].apply(os.path.isdir)]
    print(f"Number of rows with clip directory available: {len(df_anno)}")
    
    print(f"Sample annotation row:")
    print(json.dumps(df_anno.iloc[0].to_dict(), indent=4))
    
    return df_anno, df_advb, data_config


def read_frames_from_dir(clip_dir: str, num_frames: int = 8) -> torch.Tensor:
    """
    Read frames from a directory.
    
    Args:
        clip_dir: Path to directory containing frame images
        num_frames: Number of frames to sample
        
    Returns:
        Tensor of shape (T, C, H, W) with dtype uint8
    """
    paths = natsorted(glob(f"{clip_dir}/*"))
    sf = 0
    ef = len(paths)
    num_frames = min(num_frames, ef - sf)
    indices = np.linspace(sf, ef, num_frames, endpoint=False, dtype=int)
    paths = np.array(paths)[indices]
    frames = [PIL.Image.open(f).convert("RGB") for f in paths]
    x = torch.stack([torch.from_numpy(np.asarray(f)) for f in frames])
    x = x.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return x


def compute_adverb_accuracy(
    video_embeddings: Dict[str, torch.Tensor],
    text_embeddings: Dict[str, torch.Tensor],
    df_anno: pd.DataFrame,
    adverb_to_antonym: Dict[str, str],
    clip_id_col: str,
    action_col: str,
    adverb_col: str,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute adverb recognition accuracy.
    
    For each video, we check if the video is more similar to the text with the 
    correct adverb than the text with the antonym adverb.
    
    Args:
        video_embeddings: Dictionary mapping clip_id to video embeddings
        text_embeddings: Dictionary mapping "action/adverb" to text embeddings
        df_anno: Annotations dataframe
        adverb_to_antonym: Dictionary mapping adverbs to their antonyms
        clip_id_col: Column name for clip ID
        action_col: Column name for action
        adverb_col: Column name for adverb
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing accuracy metrics
    """
    correct = []
    failed = []
    
    for i in range(len(df_anno)):
        row = df_anno.iloc[i].to_dict()
        clip_id = row[clip_id_col]
        action = row[action_col]
        adverb = row[adverb_col]
        
        try:
            # Get embeddings
            zt_adverb = text_embeddings[f"{action}/{adverb}"]
            antonym = adverb_to_antonym[adverb]
            zt_antonym = text_embeddings[f"{action}/{antonym}"]
            zv = video_embeddings[clip_id]
            
            # Check if video is more similar to correct adverb than antonym
            is_correct = (zv @ zt_adverb) > (zv @ zt_antonym)
            correct.append(int(is_correct))
        except Exception as e:
            if verbose:
                print(f"Failed for index {i}, clip {clip_id}: {str(e)}")
            failed.append(i)
            continue
    
    accuracy = 100.0 * np.mean(correct) if len(correct) > 0 else 0.0
    
    results = {
        'accuracy': accuracy,
        'num_correct': len([c for c in correct if c == 1]),
        'num_samples': len(correct),
        'num_failed': len(failed),
        'total_samples': len(df_anno),
    }
    
    if verbose:
        print(f"Accuracy: {results['accuracy']:.2f}%%")
        print(f"Correct: {results['num_correct']}/{results['num_samples']}")
        print(f"Failed: {results['num_failed']}")
    
    return results


if __name__ == "__main__":
    
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to TARA model checkpoint')
    parser.add_argument('--device_map', type=str, default='auto',
                       help='Device map for model loading')
    parser.add_argument('--dataset', type=str, default='vatex-adverbs',
                       choices=['vatex-adverbs'],
                       help='Dataset to evaluate on')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames to sample from each video')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - only process 100 samples')
    args = parser.parse_args()

    # Load data
    df_anno, df_advb, data_config = load_data(dataset=args.dataset)
    
    # Create adverb to antonym mapping
    adverb_to_antonym = dict(df_advb.values)
    print(f"Number of adverb-antonym pairs: {len(adverb_to_antonym)}")
    
    # Get column names from config
    clip_id_col = data_config['clip_id_col']
    action_col = data_config['action_col']
    adverb_col = data_config['adverb_col']
    frame_dir = data_config['frame_dir']
    
    # Debug mode
    if args.debug:
        df_anno = df_anno.sample(n=min(100, len(df_anno)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: Evaluating on {len(df_anno)} samples only.")
    else:
        print(f"Evaluating on all {len(df_anno)} samples.")
    
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

    # Compute text embeddings for all action-adverb pairs
    print("Computing text embeddings for action-adverb pairs...")
    text_embeddings = {}
    
    for i in tqdm_iterator(range(len(df_anno)), desc='Computing text features'):
        row = df_anno.iloc[i].to_dict()
        action = row[action_col]
        adverb = row[adverb_col]
        
        # Embed adverb with action
        key = f"{action}/{adverb}"
        if key not in text_embeddings:
            prompt = f"The action {action} is performed {adverb}."
            with torch.no_grad():
                zt = model.encode_text(prompt).cpu().squeeze(0).float()
            zt = torch.nn.functional.normalize(zt, dim=-1)
            text_embeddings[key] = zt
        
        # Embed antonym with action
        antonym = adverb_to_antonym[adverb]
        key = f"{action}/{antonym}"
        if key not in text_embeddings:
            prompt = f"The action {action} is performed {antonym}."
            with torch.no_grad():
                zt = model.encode_text(prompt).cpu().squeeze(0).float()
            zt = torch.nn.functional.normalize(zt, dim=-1)
            text_embeddings[key] = zt
    
    print(f"Computed {len(text_embeddings)} unique text embeddings.")
    print('-' * 100)

    # Compute video embeddings
    print(f"Computing video embeddings for {len(df_anno)} videos...")
    video_embeddings = {}
    failed_videos = []
    
    for i in tqdm_iterator(range(len(df_anno)), desc='Computing video features'):
        row = df_anno.iloc[i].to_dict()
        clip_id = row[clip_id_col]
        
        try:
            # Read frames from directory
            frames = read_frames_from_dir(
                row['clip_dir'], 
                num_frames=args.num_frames
            )
            
            # Encode video
            with torch.no_grad():
                zv = model.encode_vision(frames.unsqueeze(0)).cpu().float().squeeze(0)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            video_embeddings[clip_id] = zv
            
        except Exception as e:
            print(f"Error processing {clip_id}: {str(e)}")
            failed_videos.append(clip_id)
            continue
    
    print(f"Successfully processed {len(video_embeddings)} videos.")
    if len(failed_videos) > 0:
        print(f"Failed to process {len(failed_videos)} videos.")
    print('-' * 100)
    
    # Filter dataframe to only include successfully processed videos
    df_anno = df_anno[df_anno[clip_id_col].isin(video_embeddings.keys())].reset_index(drop=True)
    print(f"Evaluating on {len(df_anno)} videos with valid embeddings.")
    print('-' * 100)
    
    # Compute accuracy
    print("Computing adverb recognition accuracy...")
    results = compute_adverb_accuracy(
        video_embeddings=video_embeddings,
        text_embeddings=text_embeddings,
        df_anno=df_anno,
        adverb_to_antonym=adverb_to_antonym,
        clip_id_col=clip_id_col,
        action_col=action_col,
        adverb_col=adverb_col,
        verbose=True,
    )
    
    print('-' * 100)
    print("Final Results:")
    print(json.dumps(results, indent=4))
    print('-' * 100)
    
    # Save results to file
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/adverbs_{args.dataset}.json"
    
    # Add metadata to results
    results['dataset'] = args.dataset
    results['model_path'] = args.model_path
    results['num_frames'] = args.num_frames
    results['debug'] = args.debug
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {result_path}")
