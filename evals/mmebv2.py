"""MMEB-v2 Video Tasks Evaluation"""
import os
import sys
import json

sys.path.append("..")
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset

from shared.utils.log import tqdm_iterator, print_update
from shared.utils.io import load_yml, load_jsonl
from modeling_tara import TARA


def gather_text_embeddings(model: TARA, texts: list) -> dict:
    """
    Compute text embeddings for a list of texts.
    
    Args:
        model: TARA model instance
        texts: List of text strings
        
    Returns:
        Dictionary mapping text to embeddings
    """
    ZT = {}
    for text in tqdm_iterator(texts, desc='Computing text embeddings'):
        with torch.no_grad():
            zt = model.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).squeeze(0).cpu().float()
        ZT[text] = zt
    return ZT


if __name__ == "__main__":
    
    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to TARA model checkpoint')
    parser.add_argument('--device_map', type=str, default='auto',
                       help='Device map for model loading')
    parser.add_argument('--model_name', type=str, default='tara',
                       help='Model name for feature file paths')
    parser.add_argument('--data_root', type=str, 
                       default='/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
                       help='Root directory for MMEB-v2 data')
    args = parser.parse_args()

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

    # ====================
    # Video Classification
    # ====================
    print("\n" + "="*100)
    print("VIDEO CLASSIFICATION")
    print("="*100 + "\n")
    
    data_root = args.data_root
    meta_config = load_yml(
        '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
    )
    
    # Load video embeddings (should be pre-computed)
    video_emb_path = f"{data_root}/features/{args.model_name}_video_embeddings_mmebv2_video_cls.pt"
    assert os.path.exists(video_emb_path), f"Video embeddings not found: {video_emb_path}"
    video_embs = torch.load(video_emb_path)
    print(f"Loaded video embeddings from {video_emb_path}")
    print('-' * 100)

    # SSv2
    ds_key = "SmthSmthV2"
    print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]
    eval_type = d['eval_type']
    data = load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
    data = pd.DataFrame(data)
    all_texts = np.unique(data.neg_text.sum())
    text_to_emb = gather_text_embeddings(model, all_texts)
    correct = []
    for j in tqdm_iterator(range(len(data)), desc='Gathering predictions'):
        row = data.iloc[j].to_dict()
        texts = row['neg_text']
        zt = torch.stack([text_to_emb[t] for t in texts])
        gt_index = texts.index(row['pos_text'])
        sim = video_embs[row['video_id']] @ zt.T
        pred_index = sim.argmax().item()
        correct.append(int(gt_index == pred_index))
    accuracy = np.mean(correct)
    accuracies = {'SmthSmthV2': np.round(accuracy * 100, 2)}

    # Other datasets
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        print_update(f"Processing {ds_key}")
        d = meta_config[ds_key]
        eval_type = d['eval_type']
        data = load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
        data = pd.DataFrame(data)
        print(f"Number of rows: {len(data)}")

        # Only keep those rows for which video embedding exists
        data = data[data.video_id.apply(lambda x: x in set(video_embs))]
        print(f"Number of rows after filtering: {len(data)}")

        zv = torch.stack([video_embs[c] for c in data.video_id.tolist()])
        texts_local = data.pos_text.unique()
        text_to_emb_local = gather_text_embeddings(model, texts_local)
        zt = torch.stack([text_to_emb_local[c] for c in data.pos_text.tolist()])

        sim = zv @ zt.T
        pred_indices = sim.argmax(dim=-1)
        pred_classes = [data.pos_text.tolist()[i] for i in pred_indices]
        accuracy = np.round((np.array(pred_classes) == np.array(data.pos_text)).mean() * 100, 2)
        accuracies[ds_key] = accuracy
        
        print_update("")
    
    mean_accuracy = np.mean([v for v in accuracies.values()])
    
    print('-' * 100)
    print("Video Classification Results:")
    print(json.dumps(accuracies, indent=2))
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    print('-' * 100)

    # ====================
    # Video Retrieval
    # ====================
    print("\n" + "="*100)
    print("VIDEO RETRIEVAL")
    print("="*100 + "\n")
    
    meta_config = load_yml(
        '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
    )
    
    # Load video embeddings (should be pre-computed)
    video_emb_path = f"{data_root}/features/{args.model_name}_video_embeddings_mmebv2_video_ret.pt"
    assert os.path.exists(video_emb_path), f"Video embeddings not found: {video_emb_path}"
    video_embs = torch.load(video_emb_path)
    print(f"Loaded video embeddings from {video_emb_path}")
    print('-' * 100)
    
    # This defines the huggingface repo and subset for each dataset
    # (repo, subset, split)
    json_paths = {
        "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
        "MSVD": ("VLM2Vec/MSVD", None, "test"),
        "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
        "YouCook2": ("lmms-lab/YouCook2", None, "val"),
        "VATEX": ("VLM2Vec/VATEX", None, "test"),
    }
    video_id_extractor = {
        "MSR-VTT": lambda x: x['video_id'],
        "MSVD": lambda x: x['video_id'],
        "DiDeMo": lambda x: x['video'].split('/')[-1].split('.')[0],
        "YouCook2": lambda x: x["id"],
        "VATEX": lambda x: x['videoID'],
    }
    video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
    captions_extractor = {
        "MSR-VTT": lambda x: [x["caption"]],
        "MSVD": lambda x: x["caption"],
        "DiDeMo": lambda x: [x["caption"]],
        "YouCook2": lambda x: [x['sentence']],
        "VATEX": lambda x: x["enCap"],
    }
    
    # Gather text embeddings for all datasets
    text_embeds = {}
    for ds_key in meta_config:
        print_update(f"Processing {ds_key}")
        d = meta_config[ds_key]
        repo, subset, split = json_paths[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        video_dir = f"{video_root}/{ds_key}/frames"
        video_ids = os.listdir(video_dir)
        assert len(video_ids) == len(df), f"Mismatch: {len(video_ids)} videos vs {len(df)} rows"
        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)
        print(json.dumps(df.iloc[0].to_dict(), indent=2))

        all_texts = [
            captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
        ]
        all_texts = np.unique(np.concatenate(all_texts))
        print(f"Total number of text captions: {len(all_texts)}")
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts)
        print_update("")

    # Compute retrieval accuracies
    ret_accs = {}
    for ds_key in meta_config:
        print_update(f"Processing {ds_key}")
        d = meta_config[ds_key]

        repo, subset, split = json_paths[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        video_dir = f"{video_root}/{ds_key}/frames"
        video_ids = os.listdir(video_dir)
        assert len(video_ids) == len(df), f"Mismatch: {len(video_ids)} videos vs {len(df)} rows"
        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

        print(json.dumps(df.iloc[0].to_dict(), indent=2))

        all_texts = [
            captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
        ]
        all_texts = np.unique(np.concatenate(all_texts))
        print(f"Total number of text captions: {len(all_texts)}")
        text_emb = text_embeds[ds_key]

        zv = torch.stack([video_embs[c] for c in df.video_id.tolist()])
        zt = torch.stack([text_emb[t] for t in all_texts])

        sim = zv @ zt.T
        pred_indices = sim.argmax(dim=-1)
        pred_captions = np.array([all_texts[i] for i in pred_indices])
        actu_captions = [captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))]
        is_correct = [int(x in y) for x, y in zip(pred_captions, actu_captions)]
        accuracy = np.round(np.mean(is_correct) * 100., 2).item()
        ret_accs[ds_key] = accuracy
        print_update("")

    mean_acc = np.mean([v for k, v in ret_accs.items()])
    mean_acc = np.round(mean_acc, 2)
    
    print('-' * 100)
    print("Video Retrieval Results:")
    print(json.dumps(ret_accs, indent=2))
    print(f"Mean accuracy: {mean_acc:.2f}")
    print('-' * 100)
    
    # Save all results
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    
    all_results = {
        'video_classification': accuracies,
        'video_classification_mean': float(mean_accuracy),
        'video_retrieval': ret_accs,
        'video_retrieval_mean': float(mean_acc),
        'model_path': args.model_path,
        'model_name': args.model_name,
    }
    
    result_path = f"{result_dir}/mmebv2_{args.model_name}.json"
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {result_path}")
