# ![](assets/tara-logo.png) TARA: *T*ext *A*dapted *R*etrieval *A*lignment for Nuanced Video Retrieval
<!-- # <img src="./assets/logo.png" width="24"> TARA: Time-Aware Retrieval Adaptation for Video Understanding -->

This repository contains inference and evaluation code for the TARA model based on the paper:
[Adapting MLLMs for Nuanced Video Retrieval](https://arxiv.org/abs/2512.13511)

<p align="center">
  <a href="https://bpiyush.github.io/tara-website/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/bpiyush/TARA" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub Code">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2512.13511" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/bpiyush/chirality-in-action" target="_blank">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" alt="Dataset on Hugging Face">
  </a>
</p>

<!-- Show arch fig in 75% of the screen and center it -->
<p align="center">
  <img src="./assets/arch.png" width="75%" alt="TARA architecture">
</p>
<!-- Add a caption with small font size and center it such that it align with image width and center it-->
<p style="text-align: left; font-size: 13px; width: 75%; display: block; margin: 0 auto;"><b>TARA Architecture:</b> We use EOL prompt to embed videos using an MLLM (Tarsier2-7B). We train the LLM weights with contrastive loss on carefully crafted hard-negatives to instill (i) temporal, (ii) negation and (iii) multimodal nuances in the embedding space.</p>


<!-- Add a Table of Contents here -->
## Table of Contents
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
  - [Data Preparation](#data-preparation)
  - [Embedding Computation](#embedding-computation)
  - [General evaluation: MMEB-V2 (Meng et al.)](#general-evaluation-mmeb-v2-meng-et-al)
- [Training](#training)
- [Citation](#citation)
- [License](#license)


## Installation & Setup

First, clone the repository:
```bash
git clone https://github.com/bpiyush/tara.git
cd tara
```


### 1. Install Git LFS (if not already installed)

Git LFS is required to download the model weights.

Please install Git LFS from https://git-lfs.github.com/.
You can refer to [this guide](https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab) for non-sudo installation.
I have not tested this guide, but it should work.

Check the installation:
```bash
git lfs --version
git lfs install
```
The output should be:
```
git-lfs/3.4.1 (GitHub; linux amd64; go 1.20.11; git 0898dcbc
Updated Git hooks.
Git LFS initialized.
```


### 2. Download the Model Weights
```bash
git clone https://huggingface.co/bpiyush/TARA /path/to/download/tara
cd TARA
```

This will download all model weights (may take a few minutes depending on your connection).

### 3. Install Dependencies

**Step 1 — Create and activate a conda environment:**
```bash
conda create -n tara python=3.10 -y
conda activate tara
```

**Step 2 — Install CUDA-specific wheels** (adjust the index URL if you need a different CUDA build):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
```

**Step 3 — Install `flash-attn`** (must be built against the torch version installed above):
```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

**Step 4 — Install TARA and all remaining dependencies:**
```bash
pip install -e /path/to/tara
```
where `/path/to/tara` is the root of this cloned repository. If you are already inside it, use `.` instead.

**(Optional) Verify the install:**
```bash
python -c "import torch, transformers, modeling_tara; print(torch.cuda.is_available(), transformers.__version__)"
```


## Quick Start

TARA is primarily designed to encode videos and texts in a joint embedding space under an MLLM.

```python
import torch
from modeling_tara import TARA

model = TARA.from_pretrained(
    "/path/to/download/tara",  # Load from current directory
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
n_params = sum(p.numel() for p in model.model.parameters())
print(f"Number of parameters: {round(n_params/1e9, 3)}B")

# Embed a video
video_path = "./assets/folding_paper.mp4"
with torch.no_grad():
    video_emb = model.encode_vision(video_path).cpu().squeeze(0).float()
print(f"Video embedding shape: {video_emb.shape}")  # torch.Size([3584])

# Embed a text
text = ['someone is folding a paper', 'cutting a paper', 'someone is folding a paper']
with torch.no_grad():
    text_emb = model.encode_text(text).cpu().float()
print(f"Text embedding shape: {text_emb.shape}")  # torch.Size([3, 3584])
```

For a more detailed demo, see the script at [demo_usage.py](demo_usage.py). You can run it:

```sh
python demo_usage.py --model_path /path/to/download/tara
```
The output should look something like this:

```sh
============================================================
TARA Model Demo
============================================================

[1/5] Loading model...
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'}
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.07it/s]
✓ Model loaded successfully!
Number of parameters: 8.291B
----------------------------------------------------------------------------------------------------

[2/5] Testing video encoding ...
From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` instance instead by default (as opposed to the legacy tuple of tuples format). If you want to keep returning the legacy format, please set `return_legacy_cache=True`.
✓ Video encoded successfully!
Video embedding shape: torch.Size([3584])
----------------------------------------------------------------------------------------------------

[3/5] Testing text encoding...
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
✓ Text encoded successfully!
Text: ['someone is folding a paper', 'cutting a paper', 'someone is unfolding a paper']
Text embedding shape: torch.Size([3, 3584])

[4/5] Computing video-text similarities...
✓ Similarities computed!
  'someone is folding a paper': 0.6488
  'cutting a paper': 0.3952
  'someone is unfolding a paper': 0.3009
----------------------------------------------------------------------------------------------------

[5/5] Testing negation example...
Image embedding shape: torch.Size([2, 3584])
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Text query:  ['an image of a cat but there is no dog in it']
Text-Image similarity: tensor([[0.5169, 0.3659]])
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
Text query:  ['an image of a cat and a dog together']
Text-Image similarity: tensor([[0.4364, 0.6004]])
----------------------------------------------------------------------------------------------------

[Bonus] Testing composed video retrieval...
Source-Target similarity with edit: 0.757888674736023

============================================================
Demo completed successfully! 🎉
============================================================
```


## Evaluation


### Data Preparation

We release the nuanced video retrieval splits used in the dataset in [data/](data/) folder.
For ease of use, we have combined all the data for (i) temporal, (ii) negation and (iii) multimodal
nuance into a single file where each entry is a video/text/video-text/image, etc.

```sh
data
├── nuanced_retrieval_inputs-test.csv # List of examples to embed (video, text, composed video-text, etc.) for test set
├── nuanced_retrieval_inputs-val.csv # List of examples to embed (video, text, composed video-text, etc.) for validation set
├── nuanced_retrieval_labels-test.json # Labels for test set
└── nuanced_retrieval_labels-val.json # Labels for validation set
```

An example input row looks like this:
```json
{
  'id': '138629', 
  'value': '138629',
  'nuance': 'time',
  'source': 'cia-ssv2',
  'modality': 'video',
}
```
where `id`is the unique identified, `value` is actual value (e.g., for a text caption, the ID can be different and value stores the actual caption), `nuance` is the type of nuance, 
`source` is the source of the example (e.g., `cia-ssv2` for SSv2), and `modality` is the modality of the example (e.g., `video` or `text`).


The coresponding label looks like this:
```json
['12055391_1.0']
```
which denotes the `id` of the text associated with the video.

Finally, set the right paths to the data directories in [evals/compute_embeddings.py](evals/compute_embeddings.py) 
based on your local setup.

### Embedding Computation

First, you need to compute the embeddings for the entire dataset. You can do this by running the following script:

```bash
python evals/compute_embeddings.py \
--model_path /path/to/download/tara \
--csv_path ./data/nuanced_retrieval_inputs-val.csv \
--model_name tara_7b
```

Then, run the script to compute retrieval metrics.

```bash
python evals/compute_metrics.py \
--model_path /path/to/download/tara \
--csv_path ./data/nuanced_retrieval_inputs-val.csv \
--lab_path ./data/nuanced_retrieval_labels-val.json \
--model_name tara_7b
```

### General evaluation: MMEB-V2 ([Meng et al.](https://arxiv.org/abs/2507.04590))

We evaluate on the video classification and video retrieval tasks in MMEB-V2 to demonstrate the generalizability of TARA.

First, compute video embeddings for MMEB-V2:

```bash
# Video classification tracks
python evals/compute_video_embeddings_mmeb.py \
  --model_path /path/to/download/tara \
  --model_name tara_7b \
  --task cls

# Video retrieval tracks
python evals/compute_video_embeddings_mmeb.py \
  --model_path /path/to/download/tara \
  --model_name tara_7b \
  --task ret
```

Then compute text embeddings:

```bash
# Classification text pools
python evals/compute_text_embeddings_mmeb.py \
  --model_path /path/to/download/tara \
  --model_name tara_7b \
  --task cls

# Retrieval text pools
python evals/compute_text_embeddings_mmeb.py \
  --model_path /path/to/download/tara \
  --model_name tara_7b \
  --task ret
```

Finally, compute MMEB-V2 metrics:

```bash
python evals/compute_metrics_mmebv2.py \
  --model_path /path/to/download/tara \
  --model_name tara_7b \
  --task all
```

By default, embeddings and metrics are saved to:

```bash
/path/to/download/tara/embs/tara_7b_video_embeddings_mmebv2_video_cls.pt
/path/to/download/tara/embs/tara_7b_video_embeddings_mmebv2_video_ret.pt
/path/to/download/tara/embs/tara_7b_text_embeddings_mmebv2_text_cls.pt
/path/to/download/tara/embs/tara_7b_text_embeddings_mmebv2_text_ret.pt
/path/to/download/tara/embs/metrics_tara_7b_mmebv2.json
```

If your MMEB-V2 data/config paths are different, override:
`--data_root`, `--video_cls_cfg`, `--video_ret_cfg`, `--video_ret_root`, `--save_dir`, and `--feat_dir`.


## Training

Coming soon: we are preparing the training code for TARA and should be available soon.

## Citation

If you use this model, please cite:
```bibtex
@article{tara2025,
  title={Adapting MLLMs for Nuanced Video Retrieval},
  author={Piyush Bagad and Andrew Zisserman},
  year={2025}
  journal={arXiv preprint arXiv:2512.13511}
}
```

```bibtex
@article{bagad2025chirality,
  title={Chirality in Action: Time-Aware Video Representation Learning by Latent Straightening},
  author={Bagad, Piyush and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2509.08502},
  year={2025}
}
```

## License

Apache 2.0