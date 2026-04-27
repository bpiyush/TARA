import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import argparse
import pandas as pd
import torch

import shared.utils as su
from modeling_tara import TARA


def load_data_video_cls(data_root: str, cfg_path: str) -> pd.DataFrame:
    """Load MMEB-v2 video-classification video metadata."""
    meta_config = su.io.load_yml(cfg_path)
    rows = []

    for ds_key in su.log.tqdm_iterator(meta_config, desc="Gathering cls video paths"):
        file_name = meta_config[ds_key]["json_name"]
        data_file = os.path.join(data_root, "video-tasks", "data", file_name)
        assert os.path.exists(data_file), f"Missing file: {data_file}"
        data = su.io.load_jsonl(data_file)

        ds_name = os.path.basename(meta_config[ds_key]["frame_root"])
        for item in data:
            video_id = item["video_id"]
            frame_dir = os.path.join(data_root, "video-tasks", "frames", ds_name, video_id)
            rows.append(
                {
                    "task": "cls",
                    "dataset": ds_key,
                    "video_id": video_id,
                    "frame_dir": frame_dir,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"])
    return df


def load_data_video_ret(cfg_path: str, video_root: str) -> pd.DataFrame:
    """Load MMEB-v2 video-retrieval video metadata via HF datasets."""
    from datasets import load_dataset

    meta_config = su.io.load_yml(cfg_path)

    # (hf_repo, subset, split)
    json_paths = {
        "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
        "MSVD": ("VLM2Vec/MSVD", None, "test"),
        "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
        "YouCook2": ("lmms-lab/YouCook2", None, "val"),
        "VATEX": ("VLM2Vec/VATEX", None, "test"),
    }

    video_id_extractor = {
        "MSR-VTT": lambda x: x["video_id"],
        "MSVD": lambda x: x["video_id"],
        "DiDeMo": lambda x: x["video"].split("/")[-1].split(".")[0],
        "YouCook2": lambda x: x["id"],
        "VATEX": lambda x: x["videoID"],
    }

    rows = []
    for ds_key in su.log.tqdm_iterator(meta_config, desc="Gathering ret video paths"):
        if ds_key not in json_paths:
            print(f"Skipping unsupported retrieval dataset key in config: {ds_key}")
            continue

        repo, subset, split = json_paths[ds_key]
        ds = load_dataset(repo, subset)[split]
        df = pd.DataFrame(ds)
        df["video_id"] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

        for _, item in df.iterrows():
            video_id = item["video_id"]
            frame_dir = os.path.join(video_root, ds_key, "frames", str(video_id))
            rows.append(
                {
                    "task": "ret",
                    "dataset": ds_key,
                    "video_id": str(video_id),
                    "frame_dir": frame_dir,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"])
    return df


def resolve_video_path(frame_dir: str) -> str:
    """Convert MMEB frame directory path to original mp4 path."""
    video_path = frame_dir.replace("video-tasks/frames", "video-tasks/videos") + ".mp4"
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tara_7b")
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "ret"])
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MMEB-V2",
        help="Root containing MMEB-V2 video-tasks/{data,frames,videos}",
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
        "--video_ret_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval",
        help="Root for retrieval frame folders (contains dataset-name/frames/<video_id>)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <model_path>/embs",
    )
    args = parser.parse_args()

    # Load video list
    if args.task == "cls":
        df = load_data_video_cls(data_root=args.data_root, cfg_path=args.video_cls_cfg)
    else:
        df = load_data_video_ret(cfg_path=args.video_ret_cfg, video_root=args.video_ret_root)

    if len(df) == 0:
        raise ValueError(f"No videos found for task={args.task}")

    df["video_path"] = df["frame_dir"].apply(resolve_video_path)
    df = df[df["video_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Found {len(df)} existing videos for task={args.task}")

    # Load model
    model = TARA.from_pretrained(
        args.model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)

    # Compute embeddings
    video_embeddings = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc="Computing MMEB-v2 video embeddings"):
        row = df.iloc[i].to_dict()
        video_id = row["video_id"]
        video_path = row["video_path"]

        try:
            zv = model.encode_vision(video_path).cpu().squeeze(0).float()
            zv = torch.nn.functional.normalize(zv, dim=-1)
            video_embeddings[video_id] = zv
        except Exception as e:
            print(f"Error computing embedding for {video_id}: {e}")
            continue

    save_dir = args.save_dir or os.path.join(args.model_path, "embs")
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.model_name}_video_embeddings_mmebv2_video_{args.task}.pt"
    save_path = os.path.join(save_dir, save_name)
    torch.save(video_embeddings, save_path)

    print(f"Saved embeddings to {save_path}")
    print(f"Saved {len(video_embeddings)} embeddings")
