import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import argparse
import numpy as np
import pandas as pd
import torch

import shared.utils as su
from modeling_tara import TARA


RET_JSON_PATHS = {
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),
}

RET_CAPTIONS_EXTRACTOR = {
    "MSR-VTT": lambda x: [x["caption"]],
    "MSVD": lambda x: x["caption"],
    "DiDeMo": lambda x: [x["caption"]],
    "YouCook2": lambda x: [x["sentence"]],
    "VATEX": lambda x: x["enCap"],
}


def gather_text_embeddings(model: TARA, texts, desc: str):
    text_to_emb = {}
    for text in su.log.tqdm_iterator(texts, desc=desc):
        with torch.no_grad():
            zt = model.encode_text(text).cpu().squeeze(0).float()
            zt = torch.nn.functional.normalize(zt, dim=-1)
        text_to_emb[text] = zt
    return text_to_emb


def compute_cls_text_embeddings(model: TARA, data_root: str, cfg_path: str):
    meta_config = su.io.load_yml(cfg_path)
    text_embeds = {}

    for ds_key in su.log.tqdm_iterator(meta_config, desc="Preparing CLS text pools"):
        data_file = os.path.join(data_root, "video-tasks", "data", meta_config[ds_key]["json_name"])
        data = pd.DataFrame(su.io.load_jsonl(data_file))

        if ds_key == "SmthSmthV2":
            # Multiple-choice options come from neg_text lists.
            all_texts = set()
            for options in data["neg_text"].tolist():
                for option in options:
                    all_texts.add(option)
            all_texts = sorted(all_texts)
        else:
            # Global classification: class names in pos_text.
            all_texts = sorted(data["pos_text"].unique().tolist())

        print(f"[CLS:{ds_key}] {len(all_texts)} unique texts")
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts, desc=f"Text embeddings {ds_key}")
    return text_embeds


def compute_ret_text_embeddings(model: TARA, cfg_path: str):
    from datasets import load_dataset

    meta_config = su.io.load_yml(cfg_path)
    text_embeds = {}

    for ds_key in su.log.tqdm_iterator(meta_config, desc="Preparing RET text pools"):
        if ds_key not in RET_JSON_PATHS:
            print(f"Skipping unsupported retrieval dataset key in config: {ds_key}")
            continue

        repo, subset, split = RET_JSON_PATHS[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])

        captions_nested = [RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict()) for i in range(len(df))]
        all_texts = np.unique(np.concatenate(captions_nested)).tolist()

        print(f"[RET:{ds_key}] {len(all_texts)} unique captions")
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts, desc=f"Text embeddings {ds_key}")
    return text_embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MMEB-v2 text embeddings with TARA")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tara_7b")
    parser.add_argument("--task", type=str, default="all", choices=["cls", "ret", "all"])
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MMEB-V2",
        help="Root containing MMEB-V2 video-tasks/data",
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
        "--save_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <model_path>/embs",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.model_path, "embs")
    os.makedirs(save_dir, exist_ok=True)

    tasks = ["cls", "ret"] if args.task == "all" else [args.task]
    pending = []
    for task in tasks:
        save_path = os.path.join(save_dir, f"{args.model_name}_text_embeddings_mmebv2_text_{task}.pt")
        if os.path.exists(save_path):
            print(f"[{task.upper()}] already exists, skipping: {save_path}")
        else:
            pending.append(task)

    if len(pending) == 0:
        print("All requested text embeddings already exist.")
        raise SystemExit(0)

    model = TARA.from_pretrained(
        args.model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)

    for task in pending:
        if task == "cls":
            text_embeds = compute_cls_text_embeddings(model, data_root=args.data_root, cfg_path=args.video_cls_cfg)
        else:
            text_embeds = compute_ret_text_embeddings(model, cfg_path=args.video_ret_cfg)

        total = sum(len(v) for v in text_embeds.values())
        print(f"Total {task} texts embedded: {total}")

        save_path = os.path.join(save_dir, f"{args.model_name}_text_embeddings_mmebv2_text_{task}.pt")
        torch.save(text_embeds, save_path)
        print(f"Saved text embeddings to {save_path}")
