"""Merge fine-tuned Tarsier2 LLM weights back into full Tarsier2 MLLM.

After `finetuning_tarsier2.py` fine-tunes only the LLM part of the Tarsier2
MLLM, this script copies those fine-tuned LLM weights into a fresh copy of the
base MLLM and saves the result as a standalone checkpoint that can be loaded
directly with `TARA.from_pretrained(...)`.
"""
import os
import sys

# Make the TARA repo root importable (modeling_tara, tarsier2, shared, ...).
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModel, AutoTokenizer

from modeling_tara import BaseModelForTARA


def _ensure_not_meta(module: torch.nn.Module, name: str) -> None:
    if any(p.is_meta for p in module.parameters()):
        raise RuntimeError(f"{name} still has meta tensors; load with full weights before merging.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base_model",
        type=str,
        required=True,
        help="Path to base Tarsier2 MLLM checkpoint.",
    )
    parser.add_argument(
        "-f",
        "--fine_tuned_model",
        type=str,
        required=True,
        help="Path to fine-tuned LLM checkpoint directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <fine_tuned_model>/merged_checkpoint.",
    )
    args = parser.parse_args()

    # Need to ensure the `tara` conda env is activated.
    command = "echo $CONDA_DEFAULT_ENV"
    output = os.popen(command).read().strip()
    if output != "tara":
        raise ValueError(
            "`tara` conda env is not activated. Please activate it and try again."
        )

    print("Loading base Tarsier2 MLLM on CPU (no device_map)...")
    mllm = BaseModelForTARA.from_pretrained(
        args.base_model,
        load_llm=False,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    mllm.model.to("cpu")

    print("Loading fine-tuned Tarsier2 LLM on CPU (no device_map)...")
    llm = AutoModel.from_pretrained(
        args.fine_tuned_model,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    llm.to("cpu")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.fine_tuned_model,
        trust_remote_code=True,
    )

    _ensure_not_meta(mllm.model.language_model.model, "base language model")
    _ensure_not_meta(llm, "fine-tuned llm")

    print("Copying weights of finetuned LLM into base MLLM language model...")
    msg = mllm.model.language_model.model.load_state_dict(llm.state_dict(), strict=True)
    print(msg)

    print("Replacing tokenizer in merged checkpoint...")
    mllm.tokenizer = llm_tokenizer
    if hasattr(mllm, "processor") and hasattr(mllm.processor, "tokenizer"):
        mllm.processor.tokenizer = llm_tokenizer

    save_dir = args.save_dir or f"{args.fine_tuned_model}/merged_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving merged model to {save_dir}")
    mllm.model.save_pretrained(save_dir)
    mllm.tokenizer.save_pretrained(save_dir)

    # Tarsier2 processor save path can be nested depending on wrapper class.
    if hasattr(mllm, "super_processor") and hasattr(mllm.super_processor, "processor"):
        mllm.super_processor.processor.save_pretrained(save_dir)
    elif hasattr(mllm, "processor"):
        try:
            mllm.processor.save_pretrained(save_dir)
        except Exception:
            if hasattr(mllm.processor, "processor") and hasattr(mllm.processor.processor, "processor"):
                mllm.processor.processor.processor.save_pretrained(save_dir)
            else:
                raise

    print(f"Saved merged model to {save_dir}")
