"""Zero-shot Action Recognition Evaluation"""
import os
import sys
import json
from typing import Dict, List

sys.path.append("..")
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_yml
from modeling_tara import TARA, read_frames_decord


def load_data(dataset: str = 'ucf101') -> pd.DataFrame:
    """
    Load dataset configuration and CSV file.
    
    Args:
        dataset: Name of dataset to load (ucf101, hmdb51, kinetics-verbs)
        
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
    csv_path = data_config['csv_path']
    video_dir = data_config['video_dir']
    ext = data_config['ext']
    id_col = data_config['id_col']
    
    # Load CSV
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    
    # Add video path column
    df['video_path'] = df[id_col].apply(lambda x: f"{video_dir}/{x}.{ext}")
    
    # Filter to only existing videos
    df = df[df.video_path.apply(os.path.exists)]
    
    print(f"Dataset: {dataset}")
    print(f"Number of rows: {len(df)}")
    print(f"Sample row:")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))
    
    return df


def compute_zeroshot_accuracy(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor, 
    true_labels: List[str],
    class_names: List[str],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute zero-shot classification accuracy.
    
    Args:
        video_embeddings: Tensor of shape (num_videos, embed_dim)
        text_embeddings: Tensor of shape (num_classes, embed_dim)
        true_labels: List of ground truth class names for each video
        class_names: List of all unique class names (in same order as text_embeddings)
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing:
        - accuracy: Top-1 accuracy as percentage
        - top5_accuracy: Top-5 accuracy as percentage (if num_classes >= 5)
    """
    assert video_embeddings.shape[0] == len(true_labels), \
        f"Number of videos {video_embeddings.shape[0]} != number of labels {len(true_labels)}"
    assert text_embeddings.shape[0] == len(class_names), \
        f"Number of text embeddings {text_embeddings.shape[0]} != number of classes {len(class_names)}"
    
    # Compute similarity scores: (num_videos, num_classes)
    similarity_scores = video_embeddings @ text_embeddings.t()
    
    # Get top-1 predictions
    pred_indices = similarity_scores.argmax(dim=-1)
    pred_classes = [class_names[i] for i in pred_indices]
    
    # Compute top-1 accuracy
    top1_correct = sum([pred_classes[i] == true_labels[i] for i in range(len(true_labels))])
    top1_accuracy = 100.0 * top1_correct / len(true_labels)
    
    results = {
        'accuracy': top1_accuracy,
        'num_samples': len(true_labels),
        'num_classes': len(class_names),
    }
    
    # Compute top-5 accuracy if there are enough classes
    if len(class_names) >= 5:
        top5_indices = similarity_scores.topk(k=5, dim=-1).indices
        top5_correct = 0
        for i, true_label in enumerate(true_labels):
            top5_preds = [class_names[idx] for idx in top5_indices[i]]
            if true_label in top5_preds:
                top5_correct += 1
        top5_accuracy = 100.0 * top5_correct / len(true_labels)
        results['top5_accuracy'] = top5_accuracy
    
    if verbose:
        print(f"Top-1 Accuracy: {results['accuracy']:.2f}%")
        if 'top5_accuracy' in results:
            print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Number of classes: {results['num_classes']}")
    
    return results


if __name__ == "__main__":
    
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True,
                       help='Path to TARA model checkpoint')
    parser.add_argument('--device_map', type=str, default='auto',
                       help='Device map for model loading')
    parser.add_argument('--dataset', type=str, default='ucf101',
                       choices=['ucf101', 'hmdb51', 'kinetics-verbs'],
                       help='Dataset to evaluate on')
    parser.add_argument('--eval_on_subset', action='store_true',
                       help='Evaluate on a subset (20%%) for debugging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - only process 200 samples')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample from each video')
    args = parser.parse_args()

    # Load data
    debug = args.debug
    eval_on_subset = args.eval_on_subset
    df = load_data(dataset=args.dataset)
    
    # Load dataset config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    all_configs = load_yml(cfg_path)
    data_config = all_configs[args.dataset]
    
    # Remove duplicates based on video ID
    id_col = data_config['id_col']
    target_col = data_config['target']
    
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    
    if eval_on_subset:
        df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        print(f"Evaluating on {len(df)} samples only (20% subset).")
    elif debug:
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
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

    # Compute text features for all classes
    classes = df[target_col].unique()
    print(f"Computing text embeddings for {len(classes)} classes...")
    text_embeddings = {}
    for class_name in tqdm_iterator(classes, desc='Computing text features'):
        with torch.no_grad():
            zt = model.encode_text(class_name).cpu().float().squeeze(0)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        text_embeddings[class_name] = zt
    
    print('-' * 100)

    # Compute video features
    print(f"Computing video embeddings for {len(df)} videos...")
    video_embeddings = {}
    failed_videos = []
    
    for i in tqdm_iterator(range(len(df)), desc='Computing video features'):
        row = df.iloc[i].to_dict()
        video_id = row[id_col]
        video_path = row['video_path']
        
        try:
            with torch.no_grad():
                zv = model.encode_vision(
                    read_frames_decord(video_path, num_frames=args.num_frames).unsqueeze(0)
                ).cpu().float().squeeze(0)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            video_embeddings[video_id] = zv
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            failed_videos.append(video_id)
            continue

    # Filter dataframe to only include successfully processed videos
    df = df[df[id_col].isin(video_embeddings.keys())].reset_index(drop=True)
    print(f"Successfully processed {len(df)} videos.")
    if len(failed_videos) > 0:
        print(f"Failed to process {len(failed_videos)} videos.")
    print('-' * 100)
    
    # Prepare data for evaluation
    # Stack embeddings in the same order as dataframe
    video_tensor = torch.stack([video_embeddings[df.iloc[i][id_col]] for i in range(len(df))])
    text_tensor = torch.stack([text_embeddings[c] for c in classes])
    true_labels = df[target_col].tolist()
    class_names = classes.tolist()
    
    # Compute accuracy
    print("Computing zero-shot classification accuracy...")
    results = compute_zeroshot_accuracy(
        video_embeddings=video_tensor,
        text_embeddings=text_tensor,
        true_labels=true_labels,
        class_names=class_names,
        verbose=True,
    )
    
    print('-' * 100)
    print("Final Results:")
    print(json.dumps(results, indent=4))
    print('-' * 100)
    
    # Save results to file
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/zsar_{args.dataset}.json"
    
    # Add metadata to results
    results['dataset'] = args.dataset
    results['model_path'] = args.model_path
    results['num_frames'] = args.num_frames
    results['eval_on_subset'] = eval_on_subset
    results['debug'] = debug
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {result_path}")
