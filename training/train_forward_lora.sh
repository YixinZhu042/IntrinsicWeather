#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-sd3.5_medium}"
LATENT_DIR="${LATENT_DIR:-latent/weatherSynthetic}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/forward_weather_synthetic_lora}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

accelerate launch --num_processes "$NUM_PROCESSES" training/train_forward_lora.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --latent_data_dir "$LATENT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mixed_precision "fp16" \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate_lora 1e-4 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --num_train_epochs 10 \
  --lora_rank 32 \
  --conditioning_dropout_prob 0.05 \
  --allow_tf32 \
  --dataloader_num_workers 4 \
  --checkpointing_steps 500 \
  --gradient_checkpointing \
  --seed 42 \
  --report_to "tensorboard" \
  --logging_dir "log"
