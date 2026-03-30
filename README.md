# IntrinsicWeather: Controllable Weather Editing in Intrinsic Space

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



## TODO
[✅] Release the paper and supplementary material  
[✅] Release WeatherSynthetic dataset  
[ ] Release WeatherReal construction pipeline  
[✅] Release pretrained checkpoints  
[✅] Add Gradio for inference  
[ ] Release training code  


---

## Dataset

### WeatherSynthetic

The synthetic dataset is on Hugging Face: **[GilgameshYX/WeatherSynthetic](https://huggingface.co/datasets/GilgameshYX/WeatherSynthetic)**.

Download:

```bash
hf download --repo-type dataset GilgameshYX/WeatherSynthetic --local-dir WeatherSynthetic_dataset
```

Layout (folder name matches `--local-dir` above):

```text
WeatherSynthetic_dataset/
├── scene.txt              # one scene name per line
├── prompt.json            # image_path → prompt (optional for some workflows)
├── Modern_city/
│   ├── image/{weather}/   # {id}_image.exr, {id}_irradiance.exr
│   └── property/          # albedo, normal, roughness, metallic
├── Small_city/
└── ...
```

**Weather types** include `sunny`, `rainy`, `foggy`, `snowy`, `overcast`, `night`, `early_morning`, `rain_storm`, `sand_storm`, and others.

**Visualization helper** (load / visualize RGB and intrinsic maps):

```bash
python -m data.WeatherSynthetic
```

---

## Environment

Create a conda env and install dependencies:

```bash
conda create -n IntrinsicWeather python=3.12 -y
conda activate IntrinsicWeather
pip install -r requirements.txt
```

### Base models

Download **Stable Diffusion 3.5 Medium** (base diffusion model):

```bash
hf download stabilityai/stable-diffusion-3.5-medium --local-dir sd3.5_medium
```

Download **DINOv2-base** (geometry / texture features for IMAA):

```bash
hf download facebook/dinov2-base --local-dir dinov2_base
```

> Install PyTorch for your CUDA/CPU from [pytorch.org](https://pytorch.org) if needed.

---

## Inverse rendering

Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| InverseRenderer-512 | [GilgameshYX/InverseRenderer-512](https://huggingface.co/GilgameshYX/InverseRenderer-512) |
| InverseRenderer-1024 | [GilgameshYX/InverseRenderer-1024](https://huggingface.co/GilgameshYX/InverseRenderer-1024) |

We provide a **512×512** model and a **1024×1024** variant; higher resolution generally improves quality.

Download (example):

```bash
hf download GilgameshYX/InverseRenderer-512 --local-dir checkpoints/InverseRenderer-512
hf download GilgameshYX/InverseRenderer-1024 --local-dir checkpoints/InverseRenderer-1024
```

### Benchmark script (`test_inverse.py`)

Runs on WeatherSynthetic, reports PSNR / SSIM / LPIPS, and saves predictions, metrics, and GT under the output directory.

> Run from the **repository root** so paths like `WeatherSynthetic_dataset/`, `sd3.5_medium/`, `checkpoints/`, and local `dino/` resolve correctly.

```bash
python test_inverse.py --output_dir inverse_output
```

### Gradio demo

> Same as above: run from the **repository root**.

```bash
python gradio_inverse_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt0 checkpoints/InverseRenderer-1024/pytorch_model-00001-of-00002.bin \
  --transformer_ckpt1 checkpoints/InverseRenderer-1024/pytorch_model-00002-of-00002.bin \
  --imaa_path checkpoints/InverseRenderer-1024/imaa.pth \
  --dino_path dino \
  --device cuda \
  --port 7860
```

---
## Forward rendering
Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| ForwardRenderer | [GilgameshYX/ForwardRenderer](https://huggingface.co/GilgameshYX/ForwardRenderer) |

Download (example):
```bash
hf download GilgameshYX/ForwardRenderer --local-dir checkpoints/ForwardRenderer
```

### Gradio demo
```bash
python gradio_forward_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt /checkpoints/ForwardRenderer/pytorch_model.bin \
  --lora_path /checkpoints/ForwardRenderer/pytorch_lora_weights.safetensors \
  --maps_dir assets/examples/intrinsic_maps \
  --device cuda \
  --port 7861
```

---

## Acknowledgements

We thank the authors of the following projects:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [RGB↔X](https://github.com/zheng95z/rgbx)
- [DiffusionRenderer (origin)](https://github.com/nv-tlabs/diffusion-renderer), [DiffusionRenderer (cosmos)](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)
