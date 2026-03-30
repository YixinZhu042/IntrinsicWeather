"""
Gradio demo: one RGB image -> five intrinsic maps (albedo, normal, roughness, metallic, irradiance).

Usage (example):
  python gradio_inverse_demo.py \\
    --sd3_path /path/to/sd3.5 \\
    --transformer_ckpt0 /path/to/pytorch_model-00001-of-00002.bin \\
    --transformer_ckpt1 /path/to/pytorch_model-00002-of-00002.bin \\
    --imaa_path imaa.pth \\
    --dino_path facebook/dinov2-base

Environment variables (optional overrides):
  SD3_PATH, INVERSE_CKPT0, INVERSE_CKPT1, IMAA_PATH, DINO_PATH
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from custom_model.transformer import SD3Transformer2DModel
from extract_dino_feature import extract_patch_tokens_min_windows
from inverse_renderer_pipeline import StableDiffusion3InstructPix2PixPipeline
from IMAA import IMAA, build_attn_mask


def _env_or(arg: str | None, key: str, default: str) -> str:
    if arg:
        return arg
    return os.environ.get(key, default)


def _load_torch(path: str, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_transformer(
    sd3_path: str,
    ckpt0: str,
    ckpt1: str,
    device: torch.device,
) -> SD3Transformer2DModel:
    transformer = SD3Transformer2DModel.from_pretrained(
        sd3_path,
        subfolder="transformer",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    in_channels = 32
    out_channels = transformer.pos_embed.proj.out_channels
    transformer.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_proj = torch.nn.Conv2d(
            in_channels,
            out_channels,
            transformer.pos_embed.proj.kernel_size,
            transformer.pos_embed.proj.stride,
            transformer.pos_embed.proj.padding,
        )
        new_proj.weight.zero_()
        new_proj.weight[:, :16, :, :].copy_(transformer.pos_embed.proj.weight)
        transformer.pos_embed.proj = new_proj

    state_dict0 = _load_torch(ckpt0, map_location="cpu")
    state_dict1 = _load_torch(ckpt1, map_location="cpu")
    state_dict = {**state_dict0, **state_dict1}
    transformer.load_state_dict(state_dict)
    transformer.eval()
    return transformer.to(device)


def pil_to_model_input(
    pil: Image.Image,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    tfm = T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
    )
    x = tfm(pil)  # [0,1], (3,H,W)
    x = x * 2.0 - 1.0
    return x.unsqueeze(0).to(device=device, dtype=dtype)


def _aov_np_to_display_uint8(output_maps: np.ndarray) -> np.ndarray:
    """Match test_inverse.py: pipeline returns float [0, 1], must scale to 0–255 before uint8."""
    x = np.asarray(output_maps)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    if x.dtype != np.uint8:
        # VaeAoVProcessor + diffusers postprocess uses [0, 1] for output_type=np
        if float(np.max(x)) <= 1.0 + 1e-3:
            x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def run_inference(
    pil_image: Image.Image | None,
    num_steps: int,
) -> list[tuple[np.ndarray, str]]:
    if pil_image is None:
        raise gr.Error("Please upload an image.")

    bundle = _GLOBAL_BUNDLE
    if bundle is None:
        raise gr.Error("Models are not loaded; check the server log for errors.")

    device: torch.device = bundle["device"]
    dtype: torch.dtype = bundle["dtype"]
    pipe = bundle["pipe"]
    imaa: IMAA = bundle["imaa"]
    dino_model = bundle["dino_model"]
    dino_processor = bundle["dino_processor"]
    im_size: int = bundle["im_size"]

    original_image = pil_to_model_input(pil_image, im_size, device, dtype)
    with torch.no_grad():
        patch_tokens = extract_patch_tokens_min_windows(
            original_image,
            dino_model,
            dino_processor,
            window_size=224,
            device=original_image.device,
        )

    output_size = (original_image.shape[2] // 16, original_image.shape[3] // 16)
    img_len = output_size[0] * output_size[1]

    prompts_list = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (lighting)",
    }
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]

    gallery: list[tuple[np.ndarray, str]] = []
    for i, aov in enumerate(required_aovs):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts_list[aov],
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=False,
        )

        with torch.no_grad():
            map_aware_mask = imaa(
                patch_tokens=patch_tokens,
                output_size=output_size,
                map_ids=torch.tensor([i], device=device),
            )
            attn_mask = build_attn_mask(map_aware_mask, 154, img_len, 0.7)

            output_maps = pipe(
                image=original_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=0,
                image_guidance_scale=0,
                num_inference_steps=int(num_steps),
                strength=1.0,
                output_type="np",
                aov=[aov],
                map_aware_mask=attn_mask.to(original_image.device),
            )[0][0]

        arr = _aov_np_to_display_uint8(output_maps)
        gallery.append((arr, prompts_list[aov]))

    return gallery


_GLOBAL_BUNDLE: dict[str, Any] | None = None


def init_models(args: argparse.Namespace) -> None:
    global _GLOBAL_BUNDLE
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device(args.device)
        dtype = torch.float16

    print(f"Loading SD3 from {args.sd3_path} ...")
    transformer = build_transformer(args.sd3_path, args.transformer_ckpt0, args.transformer_ckpt1, device)

    print(f"Loading IMAA from {args.imaa_path} ...")
    try:
        imaa_dict = torch.load(args.imaa_path, map_location=device, weights_only=False)
    except TypeError:
        imaa_dict = torch.load(args.imaa_path, map_location=device)
    imaa = IMAA(dino_model=None, processor=None, num_maps=5, map_embedding_dim=256, common_dim=128).to(device)
    imaa.load_state_dict(imaa_dict["model_state_dict"])
    imaa.eval()

    print("Building pipeline ...")
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
        args.sd3_path,
        transformer=transformer,
        use_safetensors=True,
    ).to(device, dtype=dtype)

    print(f"Loading DINO from {args.dino_path} ...")
    dino_processor = AutoImageProcessor.from_pretrained(args.dino_path)
    dino_model = AutoModel.from_pretrained(args.dino_path).to(device)
    dino_model.eval()

    _GLOBAL_BUNDLE = {
        "device": device,
        "dtype": dtype,
        "pipe": pipe,
        "imaa": imaa,
        "dino_model": dino_model,
        "dino_processor": dino_processor,
        "im_size": args.im_size,
    }
    print("Ready.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intrinsic inverse renderer — Gradio demo")
    p.add_argument("--sd3_path", type=str, default=None, help="Local SD3.5 folder (same as test_inverse.py)")
    p.add_argument("--transformer_ckpt0", type=str, default=None)
    p.add_argument("--transformer_ckpt1", type=str, default=None)
    p.add_argument("--imaa_path", type=str, default=None)
    p.add_argument("--dino_path", type=str, default=None, help="HF id or local folder for DINOv2")
    p.add_argument("--im_size", type=int, default=1024, help="Resize input to this square size")
    p.add_argument("--device", type=str, default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="gradio share link")
    p.add_argument("--num_inference_steps", type=int, default=50)
    args = p.parse_args()

    args.sd3_path = _env_or(args.sd3_path, "SD3_PATH", "sd3.5_medium")
    args.transformer_ckpt0 = _env_or(args.transformer_ckpt0, "INVERSE_CKPT0", "output_dir/pytorch_model-00001-of-00002.bin")
    args.transformer_ckpt1 = _env_or(args.transformer_ckpt1, "INVERSE_CKPT1", "output_dir/pytorch_model-00002-of-00002.bin")
    args.imaa_path = _env_or(args.imaa_path, "IMAA_PATH", "imaa.pth")
    args.dino_path = _env_or(args.dino_path, "DINO_PATH", "facebook/dinov2-base")
    return args


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.sd3_path):
        print(f"Warning: SD3 path is not a directory: {args.sd3_path}", file=sys.stderr)
    for path, label in [
        (args.transformer_ckpt0, "transformer ckpt 0"),
        (args.transformer_ckpt1, "transformer ckpt 1"),
    ]:
        if not os.path.isfile(path):
            print(f"Warning: {label} not found: {path}", file=sys.stderr)
    if not os.path.isfile(args.imaa_path):
        print(f"Warning: IMAA checkpoint not found: {args.imaa_path}", file=sys.stderr)

    init_models(args)

    default_steps = args.num_inference_steps

    with gr.Blocks(title="IntrinsicWeather — Inverse maps") as demo:
        gr.Markdown(
            "## Intrinsic decomposition\n"
            "Upload one RGB image. The model outputs **albedo**, **normal**, **roughness**, **metallic**, and **irradiance**."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Input image", height=400)
            out = gr.Gallery(
                label="Outputs (5 maps)",
                columns=5,
                rows=1,
                height=320,
                object_fit="contain",
            )
        steps = gr.Slider(10, 100, value=default_steps, step=1, label="Inference steps")
        btn = gr.Button("Run", variant="primary")

        btn.click(
            fn=run_inference,
            inputs=[inp, steps],
            outputs=out,
        )

        # Example RGB inputs: assets/examples/image/1.png ... 10.png
        _examples_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "examples", "image"
        )
        _example_rows: list[list[str]] = []
        for i in range(1, 11):
            _p = os.path.join(_examples_root, f"{i}.png")
            if os.path.isfile(_p):
                _example_rows.append([_p])
            else:
                print(f"Warning: example image missing: {_p}", file=sys.stderr)
        if _example_rows:
            gr.Examples(
                examples=_example_rows,
                inputs=[inp],
                label="Examples (1-10)",
                cache_examples=False,
            )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
