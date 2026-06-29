# [CVPR 2026 Highlight] IntrinsicWeather: Controllable Weather Editing in Intrinsic Space

<p align="center">
  <img src="assets/teaser.png" alt="IntrinsicWeather teaser" width="100%">
</p>

<p align="center">
  <a href="https://yixinzhu042.github.io/IntrinsicWeather/">Project page</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/pdf/2508.06982v6">Paper</a>
  &nbsp;·&nbsp;
  <a href="assets/supp.pdf">Supplementary</a>
</p>

<p align="center">
  <a href="https://yixinzhu042.github.io/">Yixin Zhu</a>,
  <a href="https://github.com/NK-CS-ZZL">Zuo-Liang Zhu</a>,
  <a href="https://www.patternrecognition.asia/jian/">Jian Yang</a>,
  <a href="https://miloshasan.net/">Miloš Hašan</a>,
  <a href="https://csjinxie.github.io/">Jin Xie</a>,
  <a href="https://wangningbei.github.io/">Beibei Wang</a>
</p>

---

## News

- **2026-06-29:** Training code is now open-sourced, including WeatherSynthetic latent preparation, inverse-renderer training, and forward-renderer LoRA training.

## TODO

- [x] Release the paper and supplementary material
- [x] Release WeatherSynthetic dataset
- [ ] Release WeatherReal construction pipeline
- [x] Release pretrained checkpoints
- [x] Add Gradio for inference
- [x] Release training code for WeatherSynthetic

---

## Environment

Create a conda env and install dependencies:

```bash
conda create -n IntrinsicWeather python=3.12 -y
conda activate IntrinsicWeather
pip install -r requirements.txt
```

Install PyTorch for your CUDA/CPU from [pytorch.org](https://pytorch.org) if needed.

### Base models

Download **Stable Diffusion 3.5 Medium**:

```bash
hf download stabilityai/stable-diffusion-3.5-medium --local-dir sd3.5_medium
```

Download **DINOv2-base** for IMAA / DINO patch tokens:

```bash
hf download facebook/dinov2-base --local-dir dinov2_base
```

---

## Dataset

### WeatherSynthetic

The synthetic dataset is on Hugging Face: **[GilgameshYX/WeatherSynthetic](https://huggingface.co/datasets/GilgameshYX/WeatherSynthetic)**.

Download:

```bash
hf download --repo-type dataset GilgameshYX/WeatherSynthetic --local-dir WeatherSynthetic_dataset
```

Expected layout:

```text
WeatherSynthetic_dataset/
├── scene.txt              # one scene name per line
├── prompt.json            # image_path → prompt, optional for some workflows
├── Modern_city/
│   ├── image/{weather}/   # {id}_image.exr, {id}_irradiance.exr
│   └── property/          # albedo, normal, roughness, metallic
├── Small_city/
└── ...
```

Weather types include `sunny`, `rainy`, `foggy`, `snowy`, `overcast`, `night`, `early_morning`, `rain_storm`, `sand_storm`, and others.

Visualize one sample:

```bash
python -m data.WeatherSynthetic \
  --root WeatherSynthetic_dataset \
  --output weather_synthetic_vis.png
```

---

## Training code

The training pipeline uses the released WeatherSynthetic dataset via an explicit latent-cache step.

### 1. Precompute VAE/DINO latents

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

Useful options:

- `--skip_dino`: skip `patch_tokens`; this is okay for forward-renderer LoRA training but **not** for inverse training.
- `--encode_prompts`: saves SD3 text embeddings for forward training. If omitted, `training/train_forward_lora.py` computes frozen prompt embeddings online.
- `--max_samples N`: quick debug run.
- `--overwrite`: regenerate existing HDF5 files.

### Latent scaling convention

Keep the default convention unless you intentionally modify all related code:

- `latent_image` is **scaled** by `vae.config.scaling_factor`.
- intrinsic maps (`latent_albedo`, `latent_normal`, `latent_roughness`, `latent_metallic`, `latent_irradiance`) are **unscaled** by default.

The training and inference code follow this convention consistently: render-image targets use SD latent scaling, while intrinsic-map latents are stored without multiplying by the VAE scaling factor. The inference pipelines also leave conditioning image/map latents unscaled and divide generated latents by `vae.config.scaling_factor` only before VAE decoding.

### 2. Train inverse renderer

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_inverse.sh
```

Main script:

```bash
accelerate launch training/train_inverse.py \
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

### 3. Train forward renderer LoRA

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_forward_lora.sh
```

Main script:

```bash
accelerate launch training/train_forward_lora.py \
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

See `training/README.md` for a shorter training-only checklist.

---

## Inverse rendering

Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| InverseRenderer-512 | [GilgameshYX/InverseRenderer-512](https://huggingface.co/GilgameshYX/InverseRenderer-512) |
| InverseRenderer-1024 | [GilgameshYX/InverseRenderer-1024](https://huggingface.co/GilgameshYX/InverseRenderer-1024) |

Download:

```bash
hf download GilgameshYX/InverseRenderer-512 --local-dir checkpoints/InverseRenderer-512
hf download GilgameshYX/InverseRenderer-1024 --local-dir checkpoints/InverseRenderer-1024
```

### Benchmark script

Run from the repository root:

```bash
python test_inverse.py --output_dir inverse_output
```

### Gradio demo

```bash
python gradio_inverse_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt0 checkpoints/InverseRenderer-1024/pytorch_model-00001-of-00002.bin \
  --transformer_ckpt1 checkpoints/InverseRenderer-1024/pytorch_model-00002-of-00002.bin \
  --imaa_path checkpoints/InverseRenderer-1024/imaa.pth \
  --dino_path dinov2_base \
  --device cuda \
  --port 7860
```

---

## Forward rendering

Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| ForwardRenderer | [GilgameshYX/ForwardRenderer](https://huggingface.co/GilgameshYX/ForwardRenderer) |

Download:

```bash
hf download GilgameshYX/ForwardRenderer --local-dir checkpoints/ForwardRenderer
```

### Gradio demo

```bash
python gradio_forward_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt checkpoints/ForwardRenderer/pytorch_model.bin \
  --lora_path checkpoints/ForwardRenderer/pytorch_lora_weights.safetensors \
  --maps_dir assets/examples/intrinsic_maps \
  --device cuda \
  --port 7861
```

---

## Repository layout

```text
scripts/prepare_weather_synthetic_latents.py  # EXR → VAE/DINO HDF5 cache
training/train_inverse.py                     # inverse-renderer trainer
training/train_forward_lora.py                # forward-renderer LoRA trainer
data/WeatherSynthetic.py                      # released EXR dataset loader
data/latent_datasets.py                       # HDF5 latent-cache dataset
```

Use `training/` for the WeatherSynthetic training workflow.

---

## Acknowledgements

We thank the authors of the following projects:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [RGB↔X](https://github.com/zheng95z/rgbx)
- [DiffusionRenderer (origin)](https://github.com/nv-tlabs/diffusion-renderer), [DiffusionRenderer (cosmos)](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)
