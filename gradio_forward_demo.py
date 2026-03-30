"""
Gradio demo: five intrinsic maps (albedo, normal, roughness, metallic, irradiance)
-> weather-conditioned RGB renders (reference: render_real_2.py).

Example:
  python gradio_forward_demo.py \\
    --sd3_path sd3.5_medium \\
    --transformer_ckpt checkpoints/forward_renderer/pytorch_model.bin \\
    --lora_path checkpoints/forward_renderer/pytorch_lora_weights.safetensors

Env overrides: SD3_PATH, FORWARD_TRANSFORMER_CKPT, FORWARD_LORA_PATH
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from diffusers import SD3Transformer2DModel
from forward_renderer_pipeline import StableDiffusion3InstructPix2PixPipeline


def _env_or(arg: str | None, key: str, default: str) -> str:
    if arg:
        return arg
    return os.environ.get(key, default)


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def discover_example_bases(maps_dir: str) -> list[str]:
    bases: set[str] = set()
    for f in glob.glob(os.path.join(maps_dir, "*_albedo.png")):
        bases.add(os.path.basename(f).replace("_albedo.png", ""))
    return sorted(bases, key=lambda s: (len(s), s))


REQUIRED_AOVS = ["albedo", "normal", "roughness", "metallic", "irradiance"]

WEATHER_CONDITIONS: dict[str, str] = {
    "rainy": "A rainy day.",
    "sunny": "A sunny day.",
    "snowy": "A snowy day.",
    "foggy": "A foggy day.",
}


_GLOBAL: dict[str, Any] = {}


def load_aov_image(
    pil: Image.Image | None,
    aov_name: str,
    im_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """PIL RGB -> (1,3,H,W) in [-1,1], same as render_real_2.load_aov_image."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    if pil is None:
        return torch.zeros((1, 3, im_size, im_size), device=device, dtype=dtype)

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((im_size, im_size), Image.Resampling.LANCZOS)
    t = tfm(pil)
    return t.unsqueeze(0).to(device=device, dtype=dtype)


def init_pipeline(args: argparse.Namespace) -> None:
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device(args.device)
        dtype = torch.float16

    print(f"Loading SD3 transformer from {args.sd3_path} ...")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.sd3_path,
        subfolder="transformer",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    in_channels = 96
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

    print(f"Loading transformer weights from {args.transformer_ckpt} ...")
    try:
        sd = torch.load(args.transformer_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.transformer_ckpt, map_location="cpu")
    transformer.load_state_dict(sd)
    transformer.eval()
    transformer = transformer.to(device)

    print("Building pipeline ...")
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
        args.sd3_path,
        transformer=transformer,
        use_safetensors=True,
    ).to(device, dtype=dtype)

    print(f"Loading LoRA from {args.lora_path} ...")
    pipe.load_lora_weights(args.lora_path)

    _GLOBAL.clear()
    _GLOBAL.update(
        {
            "pipe": pipe,
            "device": device,
            "dtype": dtype,
            "im_size": args.im_size,
        }
    )
    print("Ready.")


def load_example_set(
    base: str | None,
    maps_dir: str,
) -> tuple[Image.Image | None, ...]:
    """Load five PNGs for basename like image_11 from maps_dir."""
    if not base:
        return None, None, None, None, None
    paths = [os.path.join(maps_dir, f"{base}_{aov}.png") for aov in REQUIRED_AOVS]
    ims: list[Image.Image | None] = []
    for p in paths:
        if os.path.isfile(p):
            ims.append(Image.open(p).convert("RGB"))
        else:
            ims.append(None)
    return ims[0], ims[1], ims[2], ims[3], ims[4]


def run_forward(
    alb: Image.Image | None,
    nor: Image.Image | None,
    rou: Image.Image | None,
    met: Image.Image | None,
    irr: Image.Image | None,
    num_steps: float,
    guidance_scale: float,
    image_guidance_scale: float,
    seed: float,
) -> list[tuple[np.ndarray, str]]:
    bundle = _GLOBAL
    if not bundle:
        raise gr.Error("Pipeline not loaded; check the server log.")

    pipe: StableDiffusion3InstructPix2PixPipeline = bundle["pipe"]
    device: torch.device = bundle["device"]
    dtype: torch.dtype = bundle["dtype"]
    im_size: int = bundle["im_size"]

    if all(x is None for x in [alb, nor, rou, met, irr]):
        raise gr.Error("Provide at least one intrinsic map (or load an example).")

    aov_data = {
        "albedo": load_aov_image(alb, "albedo", im_size, device, dtype),
        "normal": load_aov_image(nor, "normal", im_size, device, dtype),
        "roughness": load_aov_image(rou, "roughness", im_size, device, dtype),
        "metallic": load_aov_image(met, "metallic", im_size, device, dtype),
        "irradiance": load_aov_image(irr, "irradiance", im_size, device, dtype),
    }

    gallery: list[tuple[np.ndarray, str]] = []
    seed_i = int(seed)

    with torch.no_grad():
        for wi, (weather, prompt) in enumerate(WEATHER_CONDITIONS.items()):
            gen = torch.Generator(device=device).manual_seed(seed_i + wi)
            result = pipe(
                albedo=aov_data["albedo"],
                normal=aov_data["normal"],
                roughness=aov_data["roughness"],
                metallic=aov_data["metallic"],
                irradiance=None,
                prompt=[prompt],
                guidance_scale=float(guidance_scale),
                image_guidance_scale=float(image_guidance_scale),
                num_inference_steps=int(num_steps),
                strength=1.0,
                output_type="np",
                required_aovs=REQUIRED_AOVS,
                generator=gen,
            )
            render_img = result.images[0]
            if render_img.dtype != np.uint8 and float(np.max(render_img)) <= 1.0 + 1e-3:
                render_img = render_img * 255.0
            render_img = np.clip(render_img, 0, 255).astype(np.uint8)
            gallery.append((render_img, weather))

    return gallery


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward weather renderer — Gradio demo")
    p.add_argument("--sd3_path", type=str, default=None)
    p.add_argument("--transformer_ckpt", type=str, default=None)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--maps_dir", type=str, default=None, help="Folder with *_{albedo,normal,...}.png examples")
    p.add_argument("--im_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = _repo_root()
    args.sd3_path = _env_or(args.sd3_path, "SD3_PATH", "sd3.5_medium")
    args.transformer_ckpt = _env_or(
        args.transformer_ckpt,
        "FORWARD_TRANSFORMER_CKPT",
        os.path.join(root, "checkpoints", "forward_renderer", "pytorch_model.bin"),
    )
    args.lora_path = _env_or(
        args.lora_path,
        "FORWARD_LORA_PATH",
        os.path.join(root, "checkpoints", "forward_renderer", "pytorch_lora_weights.safetensors"),
    )
    if args.maps_dir is None:
        args.maps_dir = os.path.join(root, "assets", "examples", "intrinsic_maps")
    return args


def main() -> None:
    args = parse_args()

    for path, name in [
        (args.sd3_path, "SD3"),
        (args.transformer_ckpt, "forward transformer ckpt"),
        (args.lora_path, "forward LoRA"),
    ]:
        if name == "SD3" and not os.path.isdir(path):
            print(f"Warning: SD3 path is not a directory: {path}", file=sys.stderr)
        elif name != "SD3" and not os.path.isfile(path):
            print(f"Warning: {name} not found: {path}", file=sys.stderr)

    init_pipeline(args)

    maps_dir = args.maps_dir
    example_bases = discover_example_bases(maps_dir)
    example_rows: list[list[str]] = []
    for b in example_bases:
        row = [os.path.join(maps_dir, f"{b}_{aov}.png") for aov in REQUIRED_AOVS]
        if all(os.path.isfile(p) for p in row):
            example_rows.append(row)

    default_steps = float(args.num_inference_steps)
    default_gs = float(args.guidance_scale)
    default_igs = float(args.image_guidance_scale)
    default_seed = float(args.seed)

    with gr.Blocks(title="IntrinsicWeather — Forward weather render") as demo:
        gr.Markdown(
            "## Forward weather rendering\n"
            "Upload **albedo**, **normal**, **roughness**, **metallic**, and **irradiance** maps, "
            "or load a preset from `assets/examples/intrinsic_maps`. "
            "Outputs: renders for **rainy**, **sunny**, **snowy**, **foggy**."
        )

        # Single row of inputs only: a separate Gallery duplicated these and used a small fixed height,
        # which cropped 512px maps. Use tall Image components so the full frame is visible.
        _ih = 440
        with gr.Row():
            example_dd = gr.Dropdown(
                choices=example_bases if example_bases else [],
                label="Load example set",
                value=example_bases[0] if example_bases else None,
            )
            load_btn = gr.Button("Load selected example", variant="secondary")

        with gr.Row(equal_height=True):
            in_albedo = gr.Image(type="pil", label="albedo", height=_ih)
            in_normal = gr.Image(type="pil", label="normal", height=_ih)
            in_rough = gr.Image(type="pil", label="roughness", height=_ih)
            in_metal = gr.Image(type="pil", label="metallic", height=_ih)
            in_irrad = gr.Image(type="pil", label="irradiance", height=_ih)

        with gr.Row():
            steps = gr.Slider(10, 100, value=default_steps, step=1, label="Inference steps")
            gscale = gr.Slider(1.0, 15.0, value=default_gs, step=0.5, label="guidance_scale")
            igscale = gr.Slider(0.0, 5.0, value=default_igs, step=0.1, label="image_guidance_scale")
            seed = gr.Slider(0, 2**31 - 1, value=default_seed, step=1, label="Seed (offset per weather)")

        run_btn = gr.Button("Render all weathers", variant="primary")

        out_gal = gr.Gallery(
            label="Rendered outputs",
            columns=4,
            rows=1,
            height=560,
            object_fit="contain",
        )

        def _on_load(b):
            return load_example_set(b, maps_dir)

        load_btn.click(
            fn=_on_load,
            inputs=[example_dd],
            outputs=[in_albedo, in_normal, in_rough, in_metal, in_irrad],
        )
        example_dd.change(
            fn=_on_load,
            inputs=[example_dd],
            outputs=[in_albedo, in_normal, in_rough, in_metal, in_irrad],
        )

        run_btn.click(
            fn=run_forward,
            inputs=[in_albedo, in_normal, in_rough, in_metal, in_irrad, steps, gscale, igscale, seed],
            outputs=out_gal,
        )

        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[in_albedo, in_normal, in_rough, in_metal, in_irrad],
                label="Examples (intrinsic_maps, 5 maps)",
                cache_examples=False,
            )

        if example_bases:
            demo.load(
                fn=_on_load,
                inputs=[example_dd],
                outputs=[in_albedo, in_normal, in_rough, in_metal, in_irrad],
            )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
