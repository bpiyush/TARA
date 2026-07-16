# Training TARA

TARA is trained by fine-tuning the **LLM** of the Tarsier2-7B MLLM on **triplet
contrastive loss on text-only data** using an EOL ("summarize in one word") prompt and an in-batch
contrastive (SimCSE-style) loss with hard negatives. Only the language-model
weights are updated during training; the vision tower is left untouched. After
training, the fine-tuned LLM is merged back into the full MLLM to produce a
standalone checkpoint that loads directly with `TARA.from_pretrained(...)`.

The full pipeline has three steps:

1. [Prepare the training data](#1-training-data-format) (CSV of contrastive triplets)
2. [Download the base Tarsier2 checkpoint](#2-download-the-base-tarsier2-checkpoint) and [run fine-tuning](#3-run-fine-tuning)
3. [Merge the weights](#4-merge-weights-into-the-final-model) into the final model

All scripts live in the [`training/`](training/) folder:

```sh
training
├── train_tarsier2.sh          # driver script (launches deepspeed)
├── finetuning_tarsier2.py     # main fine-tuning entrypoint (Tarsier2-specific)
├── finetuning.py              # shared helpers (trainer, collator, similarity, prompt gen)
├── merge_weights_tarsier2.py  # merges fine-tuned LLM back into the MLLM
└── ds.config                  # DeepSpeed ZeRO-2 config
```


## 1. Training data format

Training data is a **CSV file** of contrastive triplets, one example per row.
The loss pulls each anchor (`sent0`) towards its positive (`sent1`) and pushes it
away from its hard negative (`hard_neg`) as well as all other in-batch examples.

The CSV must contain the following columns:

| Column     | Required                         | Description                                                        |
| ---------- | -------------------------------- | ------------------------------------------------------------------ |
| `sent0`    | yes                              | Anchor sentence (the query text).                                  |
| `sent1`    | yes                              | Positive sentence (semantically matching / paraphrase of `sent0`). |
| `hard_neg` | yes (when `--use_neg_sentence`)  | Hard negative sentence (a subtly wrong / contrastive variant).     |

Example CSV:

```csv
sent0,sent1,hard_neg
"someone is folding a paper","a person folds a sheet of paper","someone is unfolding a paper"
"a cat sitting alone on a sofa","a lone cat rests on the couch","a cat and a dog together on a sofa"
"the man picks up the cup","a person lifts a cup","the man puts down the cup"
```

Notes:

- Each text is wrapped in the model's EOL text prompt
  (`<sent>\nSummary above sentence in one word:`) before tokenization, so you only
  need to provide the raw sentences.
- Sentences are truncated to `--cutoff_len` tokens (default `32`), so keep each
  field short (single sentences / short captions work best).
- `--use_neg_sentence` (enabled by default in the driver script) requires the
  `hard_neg` column. If you train without it, only `sent0`/`sent1` are used and
  negatives come purely from other in-batch examples.
- The loader also accepts a `datasets`-on-disk directory or a JSON file with the
  same fields (`--data_path` auto-detects `csv` vs. directory vs. JSON).

The driver script expects the CSV at `${DATA_ROOT}/${split}.csv`, where `split`
is the first positional argument (default `covr/chiral10k-covr10k`) and
`DATA_ROOT` defaults to `./data/simcse-nli`. For example, with the defaults it
reads `./data/simcse-nli/covr/chiral10k-covr10k.csv`.


## 2. Download the base Tarsier2 checkpoint

Fine-tuning starts from the base **Tarsier2-7B** MLLM (`Tarsier2ForConditionalGeneration`),
released by the Tarsier team at
[`omni-research/Tarsier2-7b-0115`](https://huggingface.co/omni-research/Tarsier2-7b-0115).

Make sure Git LFS is installed (see the [main README](README.md#1-install-git-lfs-if-not-already-installed)),
then download the checkpoint:

```bash
git clone https://huggingface.co/omni-research/Tarsier2-7b-0115 /path/to/Tarsier2-7b-0115
```

Alternatively, with the Hugging Face CLI:

```bash
hf download omni-research/Tarsier2-7b-0115 --local-dir /path/to/Tarsier2-7b-0115
```


## 3. Run fine-tuning

Fine-tuning is launched with DeepSpeed (ZeRO-2) via the driver script
[`training/train_tarsier2.sh`](training/train_tarsier2.sh). All machine-specific
paths are configurable through environment variables (with repo-relative
defaults), so nothing is hardcoded:

| Variable      | Default                              | Description                                             |
| ------------- | ------------------------------------ | ------------------------------------------------------- |
| `BASE_MODEL`  | `./checkpoints/Tarsier2-7b-0115`     | Path to the downloaded base Tarsier2 checkpoint.        |
| `DATA_ROOT`   | `./data/simcse-nli`                  | Directory containing the `${split}.csv` training files. |
| `OUTPUT_ROOT` | `./experiments`                      | Directory where experiment outputs are written.         |
| `GPUS`        | `8`                                  | Number of GPUs.                                         |
| `NUM_NODES`   | `1`                                  | Number of nodes.                                        |

The script takes two positional arguments:

```bash
bash training/train_tarsier2.sh [split] [pooling_strategy]
```

- `split` (default `covr/chiral10k-covr10k`): the training-data name; resolves to
  `${DATA_ROOT}/${split}.csv`, and is also used to build the output directory.
- `pooling_strategy` (default `last_token`): how the sentence embedding is pooled
  from the LLM hidden states. One of `last_token` or `avg_pool`.

Example run (activate the `tara` conda env first — see the
[main README](README.md#3-install-dependencies)):

```bash
conda activate tara

BASE_MODEL=/path/to/Tarsier2-7b-0115 \
DATA_ROOT=/path/to/simcse-nli \
OUTPUT_ROOT=/path/to/experiments \
GPUS=8 \
bash training/train_tarsier2.sh covr/chiral10k-covr10k last_token
```

Key training hyperparameters (set inside the driver script, matching the paper
setup) are: global batch size `768`, micro batch size `32`, `2` epochs, learning
rate `2e-5`, warmup ratio `0.1`, `cutoff_len` `32`, bf16, gradient checkpointing,
and `--use_neg_sentence` (hard negatives enabled).

The fine-tuned **LLM** checkpoint is written to:

```
${OUTPUT_ROOT}/CaRe/${base_model_name}/${split}-stepwise
```

e.g. `./experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise`. The
heavy intermediate `checkpoint-*` directories are removed automatically at the
end; the final LLM weights and tokenizer are saved to the output directory.

> Note: only the LLM is fine-tuned and saved at this stage — this is **not** yet a
> full MLLM you can load with `TARA`. Continue to the merge step below.

### Running the fine-tuning script directly

If you prefer to bypass the driver and call the entrypoint yourself:

```bash
deepspeed --num_gpus=8 --num_nodes=1 training/finetuning_tarsier2.py \
    --model_name_or_path /path/to/Tarsier2-7b-0115 \
    --data_path /path/to/simcse-nli/covr/chiral10k-covr10k.csv \
    --batch_size 768 \
    --micro_batch_size 32 \
    --num_epochs 2 \
    --warmup_ratio 0.1 \
    --learning_rate 2e-5 \
    --cutoff_len 32 \
    --output_dir /path/to/experiments/.../covr-stepwise \
    --run_name my-run \
    --pooling_strategy last_token \
    --use_neg_sentence \
    --save_steps 100000 \
    --deepspeed training/ds.config \
    --bf16 \
    --logging_steps 1 \
    --grad_checkpoint
```


## 4. Merge weights into the final model

Because fine-tuning only updates the LLM, the last step copies the fine-tuned LLM
weights back into a fresh copy of the base MLLM and saves a self-contained
checkpoint. Use [`training/merge_weights_tarsier2.py`](training/merge_weights_tarsier2.py):

```bash
conda activate tara

python training/merge_weights_tarsier2.py \
    -b /path/to/Tarsier2-7b-0115 \
    -f /path/to/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise \
    --save_dir /path/to/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise/merged_checkpoint
```

Arguments:

- `-b` / `--base_model`: path to the base Tarsier2 MLLM checkpoint (same one used
  for training).
- `-f` / `--fine_tuned_model`: path to the fine-tuned LLM checkpoint directory
  produced in step 3.
- `--save_dir` (optional): output directory. Defaults to
  `<fine_tuned_model>/merged_checkpoint`.

The script loads both models on CPU, copies the fine-tuned LLM weights into the
MLLM's language model with a strict `load_state_dict`, swaps in the fine-tuned
tokenizer, and writes the merged model + tokenizer + processor to `save_dir`.

> The merge script requires the `tara` conda env to be active (it checks
> `$CONDA_DEFAULT_ENV`).

### Load the final model

The merged checkpoint loads directly with `TARA`:

```python
import torch
from modeling_tara import TARA

model = TARA.from_pretrained(
    "/path/to/.../merged_checkpoint",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

See the [main README](README.md#quick-start) and [`demo_usage.py`](demo_usage.py)
for how to encode videos, images, and text with the resulting model, and
[Evaluation](README.md#evaluation) for reproducing the benchmark numbers.
