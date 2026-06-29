# IntrinsicWeather Training

This directory contains the WeatherSynthetic training workflow for:

- inverse renderer training: `train_inverse.py`
- forward renderer full fine-tuning: `train_forward.py`
- forward renderer LoRA fine-tuning: `train_forward_lora.py`

All trainers read the same HDF5 latent cache generated from the released WeatherSynthetic dataset.

## 0. Prerequisites

Run commands from the repository root. Prepare these directories first:

```text
WeatherSynthetic_dataset/   # downloaded WeatherSynthetic dataset
sd3.5_medium/               # Stable Diffusion 3.5 Medium
dinov2_base/                # DINOv2-base, required for inverse renderer IMAA tokens
```

Example downloads:

```bash
hf download --repo-type dataset GilgameshYX/WeatherSynthetic --local-dir WeatherSynthetic_dataset
hf download stabilityai/stable-diffusion-3.5-medium --local-dir sd3.5_medium
hf download facebook/dinov2-base --local-dir dinov2_base
```

## 1. Prepare WeatherSynthetic latents

```bash
python scripts/prepare_weather_synthetic_latents.py \
  --dataset_root WeatherSynthetic_dataset \
  --scene_list_file scene.txt \
  --sd3_path sd3.5_medium \
  --dino_path dinov2_base \
  --output_dir latent/weatherSynthetic \
  --resolution 512 \
  --map_latent_scaling unscaled \
  --image_latent_scaling scaled \
  --encode_prompts
```

Notes:

- `--encode_prompts` is optional for `train_forward.py` and `train_forward_lora.py`; if omitted, the trainers compute frozen text embeddings online.
- Do **not** use `--skip_dino` for inverse renderer training, because `train_inverse.py` needs DINO patch tokens for IMAA.
- `--skip_dino` is fine if you only train forward renderer / forward LoRA.
- Use `--max_samples N` for a quick smoke-test cache.

## 2. Train inverse renderer

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_inverse.sh
```

Equivalent direct command:

```bash
accelerate launch --num_processes 1 training/train_inverse.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/inverse_weather_synthetic \
  --mixed_precision bf16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

The inverse trainer uses `IMAA.py` and saves IMAA/transformer states under the output checkpoint directory.

## 3. Train forward renderer full model

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_forward.sh
```

Equivalent direct command:

```bash
accelerate launch --num_processes 1 training/train_forward.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/forward_weather_synthetic \
  --mixed_precision fp16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --conditioning_dropout_prob 0.05 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

This trains the expanded forward-renderer transformer directly and writes `last-checkpoint/pytorch_model.bin`.

## 4. Train forward renderer LoRA

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_forward_lora.sh
```

Equivalent direct command:

```bash
accelerate launch --num_processes 1 training/train_forward_lora.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/forward_weather_synthetic_lora \
  --mixed_precision fp16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate_lora 1e-4 \
  --lora_rank 32 \
  --conditioning_dropout_prob 0.05 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

This trains LoRA adapters for the forward renderer and saves LoRA weights under `last-checkpoint/`.

## Latent scaling convention

Keep this convention consistent across preparation, training, and inference:

- `latent_image` is **scaled** by `vae.config.scaling_factor`.
- `latent_<aov>` maps are **unscaled** by default.

The training and inference code use this convention consistently: render-image targets use SD latent scaling, while intrinsic-map latents are stored without multiplying by `vae.config.scaling_factor`.
