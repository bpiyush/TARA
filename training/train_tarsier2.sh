#!/bin/bash
#
# Fine-tune Tarsier2-7B into the TARA video-text embedding model using
# text-only contrastive data (SimCSE/NLI-style triplets: sent0, sent1, hard_neg).
#
# Paths are configurable via environment variables so nothing is hardcoded to a
# specific machine. Override any of the following before running:
#   BASE_MODEL   - path to the base Tarsier2 checkpoint (default: ./checkpoints/Tarsier2-7b-0115)
#   DATA_ROOT    - directory containing the ${split}.csv training files (default: ./data/simcse-nli)
#   OUTPUT_ROOT  - directory where experiment outputs are written (default: ./experiments)
#   GPUS         - number of GPUs (default: 8)
#   NUM_NODES    - number of nodes (default: 1)
#
# Usage:
#   bash training/train_tarsier2.sh [split] [pooling_strategy]
#   BASE_MODEL=/path/to/tara DATA_ROOT=/path/to/data bash training/train_tarsier2.sh

set -e

# Resolve directories so the script can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

split=$1
if [ -z "$split" ]; then
    split="covr/chiral10k-covr10k"
fi
echo "Using split: $split"

POOLING_STRATEGY=${2:-last_token}
echo "Using pooling strategy: $POOLING_STRATEGY"

BASE_MODEL=${BASE_MODEL:-"$REPO_ROOT/checkpoints/Tarsier2-7b-0115"}
echo "Using base model: $BASE_MODEL"

base_model_name=$(basename "$BASE_MODEL")
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/experiments"}
OUTPUT_DIR="${OUTPUT_ROOT}/CaRe/${base_model_name}/${split}-stepwise"
echo "Using output directory: $OUTPUT_DIR"
RUN_NAME=$(basename "$OUTPUT_DIR")

BATCH_SIZE=768
MICRO_BATCH_SIZE=32
SAVE_STEPS=100000
EPOCH=2
LR=2e-5
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=${GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data/simcse-nli"}
CSV_PATH="${DATA_ROOT}/${split}.csv"

echo "$BASE_MODEL"
echo "$MICRO_BATCH_SIZE $BATCH_SIZE"
echo "Training data: $CSV_PATH"
wandb online

deepspeed --num_gpus="$GPUS" --num_nodes="$NUM_NODES" "$SCRIPT_DIR/finetuning_tarsier2.py" \
        --model_name_or_path "$BASE_MODEL" \
        --data_path "$CSV_PATH" \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --warmup_ratio $WARMUP_RATIO \
        --learning_rate $LR \
        --cutoff_len $CUTOFF_LEN \
        --output_dir "$OUTPUT_DIR"  \
        --run_name "$RUN_NAME" \
        --pooling_strategy "$POOLING_STRATEGY" \
        --use_neg_sentence \
        --save_steps $SAVE_STEPS \
        --deepspeed "$SCRIPT_DIR/ds.config" \
        --bf16 \
        --logging_steps 1 \
        --grad_checkpoint


# Delete the heavy checkpoint (model is saved separately after training)
echo "Deleting the heavy checkpoint..."
rm -rf "$OUTPUT_DIR"/checkpoint-*

echo "Done!"
