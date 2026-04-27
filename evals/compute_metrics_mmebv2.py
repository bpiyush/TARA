import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import argparse
import json
import numpy as np
import pandas as pd
import torch

import shared.utils as su


RET_JSON_PATHS = {
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),
}

RET_VIDEO_ID_EXTRACTOR = {
    "MSR-VTT": lambda x: x["video_id"],
    "MSVD": lambda x: x["video_id"],
    "DiDeMo": lambda x: x["video"].split("/")[-1].split(".")[0],
    "YouCook2": lambda x: x["id"],
    "VATEX": lambda x: x["videoID"],
}

RET_CAPTIONS_EXTRACTOR = {
    "MSR-VTT": lambda x: [x["caption"]],
    "MSVD": lambda x: x["caption"],
    "DiDeMo": lambda x: [x["caption"]],
    "YouCook2": lambda x: [x["sentence"]],
    "VATEX": lambda x: x["enCap"],
}


def eval_cls(video_embs: dict, text_embeds: dict, data_root: str, cfg_path: str):
    meta_config = su.io.load_yml(cfg_path)
    results = {}

    for ds_key in su.log.tqdm_iterator(meta_config, desc="Evaluating MMEB-v2 CLS"):
        if ds_key not in text_embeds:
            print(f"Skipping {ds_key}: no text embeddings found")
            continue

        data_file = os.path.join(data_root, "video-tasks", "data", meta_config[ds_key]["json_name"])
        data = pd.DataFrame(su.io.load_jsonl(data_file))
        data["video_id"] = data["video_id"].astype(str)
        data = data[data["video_id"].isin(set(video_embs.keys()))].reset_index(drop=True)
        if len(data) == 0:
            print(f"Skipping {ds_key}: no matching video embeddings")
            continue

        ds_text_map = text_embeds[ds_key]

        if ds_key == "SmthSmthV2":
            # Multiple-choice per sample.
            correct = []
            for i in su.log.tqdm_iterator(range(len(data)), desc=f"{ds_key} predictions"):
                row = data.iloc[i].to_dict()
                options = [t for t in row["neg_text"] if t in ds_text_map]
                if row["pos_text"] not in options:
                    continue
                if len(options) == 0:
                    continue

                zq = video_embs[row["video_id"]]
                zc = torch.stack([ds_text_map[t] for t in options])
                pred_idx = (zq @ zc.T).argmax().item()
                gt_idx = options.index(row["pos_text"])
                correct.append(int(pred_idx == gt_idx))

            acc = float(np.mean(correct) * 100.0) if len(correct) > 0 else 0.0
            results[ds_key] = round(acc, 2)
            continue

        # Standard classification: nearest class text among unique labels.
        class_labels = sorted([c for c in data["pos_text"].unique().tolist() if c in ds_text_map])
        if len(class_labels) == 0:
            print(f"Skipping {ds_key}: no overlapping class label embeddings")
            continue

        zc = torch.stack([ds_text_map[c] for c in class_labels])  # [C, D]
        correct = []
        for i in su.log.tqdm_iterator(range(len(data)), desc=f"{ds_key} predictions"):
            row = data.iloc[i].to_dict()
            gt = row["pos_text"]
            if gt not in class_labels:
                continue

            zq = video_embs[row["video_id"]]  # [D]
            pred_idx = (zq @ zc.T).argmax().item()
            pred = class_labels[pred_idx]
            correct.append(int(pred == gt))

        acc = float(np.mean(correct) * 100.0) if len(correct) > 0 else 0.0
        results[ds_key] = round(acc, 2)

    if len(results) > 0:
        results["mean"] = round(float(np.mean([v for v in results.values()])), 2)
    else:
        results["mean"] = 0.0
    return results


def eval_ret(video_embs: dict, text_embeds: dict, cfg_path: str):
    from datasets import load_dataset

    meta_config = su.io.load_yml(cfg_path)
    results = {}

    for ds_key in su.log.tqdm_iterator(meta_config, desc="Evaluating MMEB-v2 RET"):
        if ds_key not in RET_JSON_PATHS:
            print(f"Skipping unsupported retrieval dataset: {ds_key}")
            continue
        if ds_key not in text_embeds:
            print(f"Skipping {ds_key}: no text embeddings found")
            continue

        repo, subset, split = RET_JSON_PATHS[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        df["video_id"] = df.apply(lambda x: str(RET_VIDEO_ID_EXTRACTOR[ds_key](x)), axis=1)
        df = df[df["video_id"].isin(set(video_embs.keys()))].reset_index(drop=True)
        if len(df) == 0:
            print(f"Skipping {ds_key}: no matching video embeddings")
            continue

        ds_text_map = text_embeds[ds_key]
        candidate_texts = sorted(ds_text_map.keys())
        if len(candidate_texts) == 0:
            continue
        zc = torch.stack([ds_text_map[t] for t in candidate_texts])  # [T, D]

        correct = []
        for i in su.log.tqdm_iterator(range(len(df)), desc=f"{ds_key} predictions"):
            row = df.iloc[i].to_dict()
            gt_texts = RET_CAPTIONS_EXTRACTOR[ds_key](row)
            zq = video_embs[row["video_id"]]

            pred_idx = (zq @ zc.T).argmax().item()
            pred_text = candidate_texts[pred_idx]
            correct.append(int(pred_text in gt_texts))

        r1 = float(np.mean(correct) * 100.0) if len(correct) > 0 else 0.0
        results[ds_key] = round(r1, 2)

    if len(results) > 0:
        results["mean"] = round(float(np.mean([v for v in results.values()])), 2)
    else:
        results["mean"] = 0.0
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMEB-v2 with precomputed TARA embeddings")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--feat_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="tara_7b")
    parser.add_argument("--task", type=str, default="all", choices=["cls", "ret", "all"])
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MMEB-V2",
    )
    parser.add_argument(
        "--video_cls_cfg",
        type=str,
        default="/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml",
    )
    parser.add_argument(
        "--video_ret_cfg",
        type=str,
        default="/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional metrics JSON path. Defaults to <feat_dir>/metrics_<model_name>_mmebv2.json",
    )
    args = parser.parse_args()

    feat_dir = args.feat_dir or (os.path.join(args.model_path, "embs") if args.model_path else None)
    if feat_dir is None:
        raise ValueError("Either --feat_dir or --model_path must be provided.")

    tasks = ["cls", "ret"] if args.task == "all" else [args.task]
    metrics = {}

    for task in tasks:
        video_path = os.path.join(feat_dir, f"{args.model_name}_video_embeddings_mmebv2_video_{task}.pt")
        text_path = os.path.join(feat_dir, f"{args.model_name}_text_embeddings_mmebv2_text_{task}.pt")
        assert os.path.exists(video_path), f"Missing video embeddings: {video_path}"
        assert os.path.exists(text_path), f"Missing text embeddings: {text_path}"

        print(f"Loading video embeddings: {video_path}")
        video_embs = torch.load(video_path)
        video_embs = {str(k): v for k, v in video_embs.items()}

        print(f"Loading text embeddings: {text_path}")
        text_embeds = torch.load(text_path)

        if task == "cls":
            metrics["cls"] = eval_cls(
                video_embs=video_embs,
                text_embeds=text_embeds,
                data_root=args.data_root,
                cfg_path=args.video_cls_cfg,
            )
        else:
            metrics["ret"] = eval_ret(
                video_embs=video_embs,
                text_embeds=text_embeds,
                cfg_path=args.video_ret_cfg,
            )

    if "cls" in metrics and "ret" in metrics:
        metrics["overall_mean"] = round((metrics["cls"]["mean"] + metrics["ret"]["mean"]) / 2.0, 2)

    save_json = args.save_json or os.path.join(feat_dir, f"metrics_{args.model_name}_mmebv2.json")
    with open(save_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMMEB-v2 metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics JSON to {save_json}")
