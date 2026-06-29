#!/usr/bin/env python
"""Precompute latents for the public IntrinsicWeather trainers.

This script converts the released WeatherSynthetic EXR dataset into compact HDF5
files consumed by ``training/train_inverse.py`` and ``training/train_forward_lora.py``.

Latent scaling convention (important):
  * rendered image latent (``latent_image``): multiplied by ``vae.config.scaling_factor``
    because it is the denoising target decoded by the SD3 VAE;
  * intrinsic map latents (``latent_albedo``/``latent_normal``/etc.): **unscaled by
    default**, matching the original training code where map latents were encoded
    without ``vae.config.scaling_factor`` and used as conditioning/targets for AOV
    prediction.

Use ``--map_latent_scaling scaled`` only if you intentionally change both training
and inference to the scaled-map convention.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from data.WeatherSynthetic import WeatherSynthetic
from extract_dino_feature import extract_patch_tokens_min_windows

AOV_NAMES = ("albedo", "normal", "roughness", "metallic", "irradiance")
AOV_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (lighting)",
}
DEFAULT_FORWARD_PROMPT = "A realistic driving scene under the specified weather condition."


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute WeatherSynthetic VAE/DINO latents.")
    parser.add_argument("--dataset_root", type=str, default="WeatherSynthetic_dataset")
    parser.add_argument("--scene_list_file", type=str, default="scene.txt")
    parser.add_argument("--prompt_json_file", type=str, default="prompt.json")
    parser.add_argument("--output_dir", type=str, default="latent/weatherSynthetic")
    parser.add_argument("--sd3_path", type=str, default="sd3.5_medium", help="SD3/SD3.5 model path containing vae/.")
    parser.add_argument("--dino_path", type=str, default="dinov2_base", help="DINOv2 path for inverse IMAA patch tokens.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="VAE encoding batch size. DINO extraction is per sample.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--map_latent_scaling", choices=("unscaled", "scaled"), default="unscaled")
    parser.add_argument("--image_latent_scaling", choices=("scaled", "unscaled"), default="scaled")
    parser.add_argument("--dino_window_size", type=int, default=224)
    parser.add_argument("--skip_dino", action="store_true", help="Do not save patch_tokens; inverse training needs them.")
    parser.add_argument(
        "--allow_hub_downloads",
        action="store_true",
        help="Allow from_pretrained() to download missing model files. Default is local files/cache only.",
    )
    parser.add_argument("--encode_prompts", action="store_true", help="Save SD3 text embeddings for forward training.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug only: stop after N samples.")
    return parser.parse_args()


def torch_dtype(name: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def safe_name(path: str) -> str:
    return path.replace(os.sep, "__").replace(".exr", "")


def encode_vae(vae: AutoencoderKL, x: torch.Tensor, scaling: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = x.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()
    if scaling == "scaled":
        latents = latents * vae.config.scaling_factor
    return latents.detach().float().cpu()


def load_text_encoders_and_tokenizers(sd3_path: str, device: torch.device, dtype: torch.dtype, local_files_only: bool):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    tok1 = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer", local_files_only=local_files_only)
    tok2 = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer_2", local_files_only=local_files_only)
    tok3 = T5TokenizerFast.from_pretrained(sd3_path, subfolder="tokenizer_3", local_files_only=local_files_only)
    enc1 = CLIPTextModelWithProjection.from_pretrained(
        sd3_path, subfolder="text_encoder", local_files_only=local_files_only
    ).to(device, dtype=dtype).eval()
    enc2 = CLIPTextModelWithProjection.from_pretrained(
        sd3_path, subfolder="text_encoder_2", local_files_only=local_files_only
    ).to(device, dtype=dtype).eval()
    enc3 = T5EncoderModel.from_pretrained(
        sd3_path, subfolder="text_encoder_3", local_files_only=local_files_only
    ).to(device, dtype=dtype).eval()
    for m in (enc1, enc2, enc3):
        m.requires_grad_(False)
    return (enc1, enc2, enc3), (tok1, tok2, tok3)


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    out = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled = out[0]
    embeds = out.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    return embeds, pooled


def _encode_prompt_with_t5(text_encoder, tokenizer, prompt, max_sequence_length, device):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    return embeds.to(dtype=text_encoder.dtype, device=device)


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length=77, device=None):
    device = device or text_encoders[0].device
    clip_embeds = []
    pooled = []
    for tokenizer, text_encoder in zip(tokenizers[:2], text_encoders[:2]):
        e, p = _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device)
        clip_embeds.append(e)
        pooled.append(p)
    clip_prompt_embeds = torch.cat(clip_embeds, dim=-1)
    pooled_prompt_embeds = torch.cat(pooled, dim=-1)
    t5_prompt_embed = _encode_prompt_with_t5(text_encoders[-1], tokenizers[-1], prompt, max_sequence_length, device)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds.detach().float().cpu(), pooled_prompt_embeds.detach().float().cpu()


def main():
    args = parse_args()
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "True")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    prompt_json = args.prompt_json_file
    if prompt_json and not os.path.isfile(os.path.join(args.dataset_root, prompt_json)):
        prompt_json = None

    dataset = WeatherSynthetic(
        args.dataset_root,
        args.scene_list_file,
        imWidth=args.resolution,
        imHeight=args.resolution,
        prompt_json_file=prompt_json,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.sd3_path, subfolder="vae", local_files_only=not args.allow_hub_downloads
    ).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)

    dino_processor = dino_model = None
    if not args.skip_dino:
        dino_processor = AutoImageProcessor.from_pretrained(
            args.dino_path, local_files_only=not args.allow_hub_downloads
        )
        dino_model = AutoModel.from_pretrained(
            args.dino_path, local_files_only=not args.allow_hub_downloads
        ).to(device).eval()
        dino_model.requires_grad_(False)

    text_encoders = tokenizers = None
    if args.encode_prompts:
        text_encoders, tokenizers = load_text_encoders_and_tokenizers(
            args.sd3_path, device, dtype, local_files_only=not args.allow_hub_downloads
        )

    meta = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "sd3_path": args.sd3_path,
        "dino_path": None if args.skip_dino else args.dino_path,
        "resolution": args.resolution,
        "image_latent_scaling": args.image_latent_scaling,
        "map_latent_scaling": args.map_latent_scaling,
        "vae_scaling_factor": float(vae.config.scaling_factor),
        "contains_patch_tokens": not args.skip_dino,
        "contains_prompt_embeddings": bool(args.encode_prompts),
        "aov_names": list(AOV_NAMES),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    for idx in tqdm(range(total), desc="precomputing latents"):
        sample = dataset[idx]
        src_path = dataset.samples[idx]["image"]
        rel = os.path.relpath(src_path, args.dataset_root)
        out_path = out_dir / f"{safe_name(rel)}.h5"
        if out_path.exists() and not args.overwrite:
            continue

        image = sample["im"].unsqueeze(0)
        latent_image = encode_vae(vae, image, args.image_latent_scaling, device, dtype)[0]

        latent_maps = {}
        for name in AOV_NAMES:
            latent_maps[name] = encode_vae(
                vae, sample[name].unsqueeze(0), args.map_latent_scaling, device, dtype
            )[0]

        patch_tokens = None
        if not args.skip_dino:
            # DINO expects display-like RGB. Convert dataset tensor [-1,1] -> [0,1].
            dino_image = (image.float() * 0.5 + 0.5).clamp(0, 1).to(device)
            patch_tokens = extract_patch_tokens_min_windows(
                dino_image,
                dino_model,
                dino_processor,
                window_size=args.dino_window_size,
                device=device,
            )[0].detach().float().cpu()

        prompt = sample.get("prompt", "") or DEFAULT_FORWARD_PROMPT
        prompt_embeds = pooled = None
        if args.encode_prompts:
            prompt_embeds, pooled = encode_prompt(text_encoders, tokenizers, [prompt], device=device)
            prompt_embeds = prompt_embeds[0]
            pooled = pooled[0]

        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("latent_image", data=latent_image.numpy(), compression="gzip")
            for name, lat in latent_maps.items():
                h5.create_dataset(f"latent_{name}", data=lat.numpy(), compression="gzip")
            if patch_tokens is not None:
                h5.create_dataset("patch_tokens", data=patch_tokens.numpy(), compression="gzip")
            if prompt_embeds is not None:
                h5.create_dataset("encoder_hidden_states", data=prompt_embeds.numpy(), compression="gzip")
                h5.create_dataset("pooled_projection", data=pooled.numpy(), compression="gzip")
            h5.create_dataset("roughness_flag", data=np.array(1, dtype=np.int64))
            h5.create_dataset("metallic_flag", data=np.array(1, dtype=np.int64))
            h5.create_dataset("irradiance_flag", data=np.array(1, dtype=np.int64))
            h5.attrs["source_image"] = rel
            h5.attrs["image_latent_scaling"] = args.image_latent_scaling
            h5.attrs["map_latent_scaling"] = args.map_latent_scaling
            h5.attrs["vae_scaling_factor"] = float(vae.config.scaling_factor)
            h5.attrs["prompt"] = prompt

    print(f"Done. Latent cache written to: {out_dir}")
    print("Latent scaling: image=", args.image_latent_scaling, "maps=", args.map_latent_scaling)


if __name__ == "__main__":
    main()
