# IntrinsicWeather training code

This directory contains the WeatherSynthetic training workflow. The trainers use the latent cache generated from the released WeatherSynthetic dataset.

## 1. Prepare latents

```bash
python scripts/prepare_weather_synthetic_latents.py \
  --dataset_root WeatherSynthetic_dataset \
  --sd3_path sd3.5_medium \
  --dino_path dinov2_base \
  --output_dir latent/weatherSynthetic \
  --resolution 512 \
  --map_latent_scaling unscaled \
  --image_latent_scaling scaled \
  --encode_prompts
```

`--encode_prompts` is optional for `train_forward.py` and `train_forward_lora.py`; if omitted, the trainer computes frozen text embeddings online.

## 2. Train inverse renderer

```bash
LATENT_DIR=latent/weatherSynthetic MODEL_NAME=sd3.5_medium bash training/train_inverse.sh
```

## 3. Train forward renderer

```bash
LATENT_DIR=latent/weatherSynthetic MODEL_NAME=sd3.5_medium bash training/train_forward.sh
```

## 4. Train forward renderer LoRA

```bash
LATENT_DIR=latent/weatherSynthetic MODEL_NAME=sd3.5_medium bash training/train_forward_lora.sh
```

## Latent scaling convention

- `latent_image` is **scaled** by `vae.config.scaling_factor`.
- `latent_<aov>` maps are **unscaled** by default.

The training and inference code use this convention consistently: render-image targets use SD latent scaling, while intrinsic-map latents are stored without multiplying by `vae.config.scaling_factor`. Keep this convention unless you also update training and inference consistently.
