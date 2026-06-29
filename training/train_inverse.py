#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import copy
import itertools
import logging
import math
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from packaging import version
from einops import repeat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.utils.checkpoint
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import (
    # SD3Transformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)
from custom_model.transformer import SD3Transformer2DModel
from custom_model.atten_processor import MapAwareAttnProcessor2_0
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available
from data.latent_datasets import LatentDataset, split_dataset
# from transformers import ViTImageProcessor, ViTForImageClassification
from safetensors import safe_open
from torch.utils.data.distributed import DistributedSampler
from MAA import MAA, build_attn_mask
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Optional internal optimizer-recovery utilities are intentionally omitted from the public release.

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

AOV_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (lighting)",
}



def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    if "large" in base_model:
        model_variant = "SD3.5-Large"
        license_url = "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md"
        variant_tags = ["sd3.5-large", "sd3.5", "sd3.5-diffusers"]
    else:
        model_variant = "SD3"
        license_url = "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md"
        variant_tags = ["sd3", "sd3-diffusers"]

    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {model_variant} DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [SD3 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sd3.md).

Was the text encoder fine-tuned? {train_text_encoder}.

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained('{repo_id}', torch_dtype=torch.float16).to('cuda')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

## License

Please adhere to the licensing terms as described `[here]({license_url})`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "template:sd-lora",
    ]
    tags += variant_tags

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        local_files_only=not args.allow_hub_downloads,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        local_files_only=not args.allow_hub_downloads,
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=args.revision,
        variant=args.variant,
        local_files_only=not args.allow_hub_downloads,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        local_files_only=not args.allow_hub_downloads,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

class MapAwareDiT(nn.Module):
    def __init__(self, maa, transformer):
        super().__init__()
        self.maa = maa
        self.transformer = transformer
    
    @property
    def device(self):
        """Get the device of the transformer (which should be the same as maa after accelerator.prepare)"""
        return next(self.transformer.parameters()).device

    def forward(self, patch_tokens, map_aware_mask_size, map_ids, hidden_states, encoder_hidden_states, timestep, pooled_projections):

        map_aware_mask = self.maa(patch_tokens=patch_tokens, output_size=map_aware_mask_size, map_ids=map_ids)
        # print("map_aware_mask.shape:", map_aware_mask.shape)

        attn_mask = build_attn_mask(map_aware_mask, encoder_hidden_states.shape[1], hidden_states.shape[-2] * hidden_states.shape[-1], 0.7)

        trans_out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            map_aware_mask=attn_mask,
            return_dict=False,
        )[0]
        return map_aware_mask, trans_out

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--latent_data_dir",
        type=str,
        default=None,
        help=(
            "Directory created by scripts/prepare_weather_synthetic_latents.py. "
            "It should contain HDF5 files with latent_image/latent_<aov> keys. "
            "If omitted, WEATHER_SYNTHETIC_LATENT_DIR or ./latent/weatherSynthetic is used."
        ),
    )
    parser.add_argument(
        "--scene_list_file",
        type=str,
        default=None,
        help="Optional scene list for latent datasets that preserve WeatherSynthetic scene folders.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.95,
        help="Fraction of the open WeatherSynthetic latent dataset used for training.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--allow_hub_downloads",
        action="store_true",
        help="Allow from_pretrained() to download missing files from Hugging Face Hub. By default local paths/caches are used only.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    # parser.add_argument(
    #     "--no_of_classes",
    #     type=int,
    #     default=77,
    #     help=(
    #         "number of classes which op_embedings"
    #     ),
    # )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--v1_checkpoint_path",
        type=str,
        default=None,
        help="Path to v1 checkpoint for loading model weights and optimizer state. Used when resuming from v1 model.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

# def save_sharded_checkpoint(accelerator: Accelerator, model, save_dir: str, tag: str = None):
#     """
#     安全分片保存 ZeRO-2 模型。
#     每张 GPU 保存自己 shard，避免 ALLREDUCE timeout。
    
#     Args:
#         accelerator: accelerate 加速器对象
#         model: 已经通过 accelerator.prepare 的模型
#         save_dir: checkpoint 根目录
#         tag: 可选，checkpoint 标记，如 step100
#     """
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # 当前 rank 文件名
#     local_rank = accelerator.local_process_index
#     file_name = f"model_rank{local_rank}.pt"
#     if tag:
#         file_name = f"{tag}_{file_name}"
#     shard_file = save_dir / file_name

#     # unwrap ZeRO 模型，获取原始模型
#     unwrapped_model = accelerator.unwrap_model(model)

#     # 获取本 rank 权重
#     state_dict = unwrapped_model.state_dict()

#     # 保存 shard，safe_serialization=True 可以避免 pickle 跨平台问题
#     torch.save(state_dict, shard_file, _use_new_zipfile_serialization=True)
    
#     if accelerator.is_main_process:
#         print(f"[INFO] Saved shard for rank {local_rank} -> {shard_file}")

def main(args):
    if args.train_text_encoder:
        raise ValueError("The public IntrinsicWeather trainers freeze text encoders; please omit --train_text_encoder.")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # # ==========================================================
    # # 在这里添加检查代码
    # # ==========================================================
    # from accelerate.utils import DistributedType

    # logger.info("================== ACCELERATOR STATE CHECK ==================")
    # logger.info(f"Distributed type: {accelerator.state.distributed_type}")
    # logger.info(f"Number of processes: {accelerator.state.num_processes}")
    
    # if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
    #     logger.info("DeepSpeed is enabled!")
    #     # 您甚至可以打印出 Accelerate 解析后的 DeepSpeed 配置
    #     logger.info(f"DeepSpeed plugin config: {accelerator.state.deepspeed_plugin.deepspeed_config}")
    # else:
    #     logger.info("DeepSpeed is NOT enabled.")
    # logger.info("===========================================================")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
                local_files_only=not args.allow_hub_downloads,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            free_memory()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        local_files_only=not args.allow_hub_downloads,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        local_files_only=not args.allow_hub_downloads,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        local_files_only=not args.allow_hub_downloads,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=not args.allow_hub_downloads
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", local_files_only=not args.allow_hub_downloads
    ).to(accelerator.device)

    # load weather_class_embedder with timestep_embedder
    # 'time_text_embed.timestep_embedder.linear_1.bias', 
    # 'time_text_embed.timestep_embedder.linear_1.weight', 
    # 'time_text_embed.timestep_embedder.linear_2.bias', 
    # 'time_text_embed.timestep_embedder.linear_2.weight',
    # Example (local debugging): inspect keys in diffusion_pytorch_model.safetensors with safe_open
    # file_path = "<your_path>/diffusion_pytorch_model.safetensors"
    # with safe_open(file_path, framework="pt", device="cpu") as f:
    #     # 获取所有键
    #     keys = f.keys()
    #     print("Keys in safetensors file:", keys)

    #     # 读取特定键的值
    #     weight = f.get_tensor("time_text_embed.timestep_embedder.linear_1.bias")
    #     print("Weight shape:", weight.shape)

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        revision=args.revision, 
        variant=args.variant,
        low_cpu_mem_usage=False,
        device_map=None,
        local_files_only=not args.allow_hub_downloads,
    )

    transformer.set_attn_processor(MapAwareAttnProcessor2_0())

    transformer.requires_grad_(True)
    vae.requires_grad_(False)

    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
        text_encoder_two.requires_grad_(True)
        text_encoder_three.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
    
    logger.info("Initializing the SD3.5 InstructPix2Pix DiT from the pretrained DiT.")
    in_channels = 32
    out_channels = transformer.pos_embed.proj.out_channels
    transformer.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_proj = nn.Conv2d(
            in_channels, out_channels,transformer.pos_embed.proj.kernel_size, transformer.pos_embed.proj.stride, transformer.pos_embed.proj.padding
        )
        new_proj.weight.zero_()
        new_proj.weight[:, :16, :, :].copy_(transformer.pos_embed.proj.weight)
        transformer.pos_embed.proj = new_proj

    maa = MAA(dino_model=None, processor=None, num_maps=5, map_embedding_dim=256, common_dim=128)
    # Example: maa_ckpt = torch.load(os.environ["MAA_CHECKPOINT"], map_location="cpu")
    # print(maa_ckpt.keys())
    # maa.load_state_dict(maa_ckpt['model_state_dict'])
    maa.train()
    MAAtransformer = MapAwareDiT(maa=maa, transformer=transformer)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if not args.train_text_encoder:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        MAAtransformer.transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
            text_encoder_three.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # def save_model_hook(models, weights, output_dir):
    #     if accelerator.is_main_process:
    #         for i, model in enumerate(models):
    #             if isinstance(unwrap_model(model), SD3Transformer2DModel):
    #                 unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
    #             elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
    #                 if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
    #                     hidden_size = unwrap_model(model).config.hidden_size
    #                     if hidden_size == 768:
    #                         unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
    #                     elif hidden_size == 1280:
    #                         unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
    #                 else:
    #                     unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
    #             else:
    #                 raise ValueError(f"Wrong model supplied: {type(model)=}.")

    #             # make sure to pop weight so that corresponding model is not saved again
    #             weights.pop()

    # def load_model_hook(models, input_dir):
    #     for _ in range(len(models)):
    #         # pop models so that they are not loaded again
    #         model = models.pop()

    #         # load diffusers style into model
    #         if isinstance(unwrap_model(model), SD3Transformer2DModel):
    #             load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
    #             model.register_to_config(**load_model.config)

    #             model.load_state_dict(load_model.state_dict())
    #         elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
    #             try:
    #                 load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
    #                 model(**load_model.config)
    #                 model.load_state_dict(load_model.state_dict())
    #             except Exception:
    #                 try:
    #                     load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
    #                     model(**load_model.config)
    #                     model.load_state_dict(load_model.state_dict())
    #                 except Exception:
    #                     try:
    #                         load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
    #                         model(**load_model.config)
    #                         model.load_state_dict(load_model.state_dict())
    #                     except Exception:
    #                         raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
    #         else:
    #             raise ValueError(f"Unsupported model found: {type(model)=}")

    #         del load_model

    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": MAAtransformer.parameters(), "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_encoder_one.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_two_with_lr = {
            "params": text_encoder_two.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_three_with_lr = {
            "params": text_encoder_three.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
            text_parameters_two_with_lr,
            text_parameters_three_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]


    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate
            params_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation (open-source release).
    # The original internal script mixed InteriorVerse/Hypersim/private driving data.
    # This public trainer consumes the released WeatherSynthetic latent cache only.
    _data_root = os.environ.get("INTRINSIC_WEATHER_DATA_ROOT", ".")
    latent_data_dir = args.latent_data_dir or os.environ.get(
        "WEATHER_SYNTHETIC_LATENT_DIR",
        os.path.join(_data_root, "latent", "weatherSynthetic"),
    )
    scene_list_file = args.scene_list_file or os.environ.get("WEATHER_SYNTHETIC_SCENE_LIST")
    dataset = LatentDataset(
        latent_data_dir,
        scene_list_file=scene_list_file,
        include_patch_tokens=True,
        include_prompt_embeds=False,
    )

    train_dataset, _ = split_dataset(dataset, train_ratio=args.train_split, seed=args.seed)

    logger.info(f"Using WeatherSynthetic latent cache: {latent_data_dir}")
    logger.info(
        "Expected latent convention: image latents are VAE-scaled; intrinsic map latents are unscaled "
        "(see scripts/prepare_weather_synthetic_latents.py --map_latent_scaling)."
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        pin_memory=True
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds



    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        MAAtransformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            MAAtransformer, optimizer, train_dataloader, lr_scheduler
        )
        
        # Debug info after accelerator.prepare
        if accelerator.is_main_process:
            logger.info(f"MAAtransformer after prepare: {type(MAAtransformer)}")
            logger.info(f"MAAtransformer device after prepare: {MAAtransformer.device}")
            logger.info(f"MAA module device: {next(MAAtransformer.maa.parameters()).device}")
            logger.info(f"Transformer module device: {next(MAAtransformer.transformer.parameters()).device}")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    initial_global_step = 0
    
    if args.v1_checkpoint_path:
        raise ValueError(
            "--v1_checkpoint_path depended on an internal optimizer-recovery helper and is not supported "
            "in the public trainer. Use --resume_from_checkpoint for checkpoints produced by this script."
        )

    if args.resume_from_checkpoint:
        # 原有的v2 checkpoint恢复逻辑
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_global_step = global_step * args.gradient_accumulation_steps
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts_list = AOV_PROMPTS
    # pre-calculate prompt embeds to save memory
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    for prompts in prompts_list.values():
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                            [prompts]*args.train_batch_size, text_encoders, tokenizers
                        )
        # print(prompt_embeds.shape, pooled_prompt_embeds.shape)
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    
    del tokenizer_one, tokenizer_two, tokenizer_three, text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizers, text_encoders
    free_memory()
    torch.cuda.empty_cache()

    for epoch in range(first_epoch, args.num_train_epochs):
        MAAtransformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            text_encoder_three.train()

        for step, batch in enumerate(train_dataloader):
          
            models_to_accumulate = MAAtransformer
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two, text_encoder_three])
            with accelerator.accumulate(models_to_accumulate):
                original_image_embeds = batch["latent_image"].to(MAAtransformer.device)
                # with torch.no_grad():
                #     original_image_embeds = vae.encode(batch["im"].to(transformer.device)).latent_dist.sample()
                # image = batch["im"].to(transformer.device)

                for i, cur_aov_name in enumerate(required_aovs):
                    # with torch.no_grad():
                    #     latents = vae.encode(batch[cur_aov_name].to(transformer.device)).latent_dist.sample() * vae.config.scaling_factor

                    latents = batch["latent_" + cur_aov_name].to(accelerator.device, dtype=weight_dtype)
                    valid_mask = torch.ones((latents.shape[0]), dtype=int)
                    if cur_aov_name == "roughness":
                        valid_mask = batch["roughness_flag"]  # extract the index for those valid
                    if cur_aov_name == "metallic":
                        valid_mask = batch["metallic_flag"]
                    if cur_aov_name == "irradiance":
                        valid_mask = batch["irradiance_flag"]
                    
                    if not isinstance(valid_mask, torch.Tensor):
                        valid_mask = torch.tensor(valid_mask, device=latents.device).to(torch.bool)
                    else:
                        valid_mask = valid_mask.to(device=latents.device, dtype=torch.bool)
                    # print(cur_aov_name, "valid_mask: ", valid_mask.shape, valid_mask)


                    # Sample noise that we'll add to the latents
                    
                    # print(mask.shape)
                    noise = torch.randn_like(latents) 
                    bsz = latents.shape[0]


                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    # same timesteps foe different maps of the same image
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                    # print("timesteps:", timesteps.shape, timesteps.dtype)  # torch.Size([12])

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise


                    # concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=2).to(dtype=weight_dtype)
                    # print("concatenated_noisy_latents:", concatenated_noisy_latents.shape, concatenated_noisy_latents.dtype)  # torch.Size([12, 5, 16, 64, 64])
                    # pooled_projections = pooled_prompt_embeds_list[i].repeat(bsz,1)
                    # print("encoder_hidden_states:", prompt_embeds_list[i].shape)
                    # print("pooled_projections:",pooled_prompt_embeds_list[i].shape)
                    # print("concatenated_noisy_latents:", concatenated_noisy_latents.shape)
                    # print("timesteps:", timesteps.shape)

                    
                    # 连接输入
                    concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1).to(dtype=weight_dtype)
                    patch_tokens = batch["patch_tokens"].to(accelerator.device, dtype=weight_dtype)

                    # Predict the noise residual
                    if not args.train_text_encoder: 
                        map_aware_mask_size = ((concatenated_noisy_latents.shape[-1] // MAAtransformer.transformer.config.patch_size), (concatenated_noisy_latents.shape[-2] // MAAtransformer.transformer.config.patch_size))
                        # print("map_aware_mask_size:", map_aware_mask_size)

                        map_aware_mask, model_pred = MAAtransformer(
                        patch_tokens=patch_tokens,
                        map_aware_mask_size=map_aware_mask_size,
                        map_ids=torch.tensor([i]*bsz).to(accelerator.device),
                        hidden_states=concatenated_noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_list[i],
                        pooled_projections=pooled_prompt_embeds_list[i],
                        )

                    else:
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one, tokens_two, tokens_three],
                        )
                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                    # print("model_pred_1: ", model_pred.shape)  # torch.Size([60, 16, 64, 64])

                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    # print("sigmas:", sigmas.shape, sigmas.dtype)  

                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_latents

                    
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

    
                    target = latents
                    

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()

                    # Compute regular loss.
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    # print("loss:", loss.shape, loss.dtype)  
        
                    valid_loss = loss[valid_mask]
                    if valid_loss.shape[0] == 0:
                        valid_loss = loss[0]
                    # print("valid_loss:", valid_loss.shape, valid_loss.dtype)  # torch.Size([B])
                    loss = valid_loss.mean()

                    if args.with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                MAAtransformer.parameters(),
                                text_encoder_one.parameters(),
                                text_encoder_two.parameters(),
                                text_encoder_three.parameters(),
                            )
                            if args.train_text_encoder
                            else MAAtransformer.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                lr_scheduler.step()
                
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path,  safe_serialization=False)
                    # save_sharded_checkpoint(accelerator, transformer, save_dir=args.output_dir, tag=f"checkpoint-{global_step}")
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_steps == 0:
                # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
                        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
                    )
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)
                    text_encoder_three.to(weight_dtype)
                pipeline = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    text_encoder_3=accelerator.unwrap_model(text_encoder_three),
                    transformer=accelerator.unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # pipeline_args = {"prompt": args.validation_prompt}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )
                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two, text_encoder_three
                    free_memory()

    # Save the lora layers
    accelerator.wait_for_everyone()
    
    # Add debug info before saving
    if accelerator.is_main_process:
        logger.info("Starting final checkpoint save...")
        logger.info(f"MAAtransformer device: {MAAtransformer.device}")
        logger.info(f"MAAtransformer type: {type(MAAtransformer)}")
    
    save_path = os.path.join(args.output_dir, "last-checkpoint")
    
    try:
        accelerator.save_state(save_path, safe_serialization=False)
        logger.info(f"Saved state to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        # Try alternative saving method
        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            try:
                # Save MAAtransformer components separately
                unwrapped_model = unwrap_model(MAAtransformer)
                torch.save({
                    'maa_state_dict': unwrapped_model.maa.state_dict(),
                    'transformer_state_dict': unwrapped_model.transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                }, os.path.join(save_path, "manual_checkpoint.pth"))
                logger.info(f"Saved manual checkpoint to {save_path}")
            except Exception as e2:
                logger.error(f"Manual save also failed: {e2}")
                # Last resort: save just the state dicts
                try:
                    torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model_only.pth"))
                    logger.info(f"Saved model state dict only to {save_path}")
                except Exception as e3:
                    logger.error(f"Final save attempt failed: {e3}")
    
    logger.info("End training!")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)