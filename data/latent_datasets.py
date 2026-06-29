"""Datasets for precomputed IntrinsicWeather VAE/DINO latent caches.

The public training scripts use this module instead of the internal data loaders
from the original research code.  Each sample is stored as one HDF5 file (or as
HDF5 files inside scene subdirectories) and contains VAE latents for the rendered
image plus five intrinsic maps.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset, random_split

AOV_NAMES: Tuple[str, ...] = ("albedo", "normal", "roughness", "metallic", "irradiance")


class LatentDataset(Dataset):
    """Load precomputed latents from ``scripts/prepare_weather_synthetic_latents.py``.

    Expected keys per HDF5 file:
      - ``latent_image``
      - ``latent_albedo``, ``latent_normal``, ``latent_roughness``, ``latent_metallic``, ``latent_irradiance``
      - optional ``patch_tokens`` for inverse training
      - optional ``encoder_hidden_states`` and ``pooled_projection`` for forward training
      - optional scalar validity flags: ``roughness_flag``, ``metallic_flag``, ``irradiance_flag``

    The loader deliberately keeps the key names used by the original training
    loops, so the public trainers remain close to the research implementation.
    """

    def __init__(
        self,
        latent_dir: str,
        scene_list_file: Optional[str] = None,
        include_patch_tokens: bool = False,
        include_prompt_embeds: bool = False,
        require_prompt_embeds: bool = False,
    ) -> None:
        self.latent_dir = os.path.abspath(latent_dir)
        self.include_patch_tokens = include_patch_tokens
        self.include_prompt_embeds = include_prompt_embeds
        self.require_prompt_embeds = require_prompt_embeds

        if not os.path.isdir(self.latent_dir):
            raise FileNotFoundError(
                f"Latent directory not found: {self.latent_dir}\n"
                "Run scripts/prepare_weather_synthetic_latents.py first, or pass --latent_data_dir."
            )

        self.files = self._discover_files(scene_list_file)
        if not self.files:
            raise FileNotFoundError(f"No .h5/.hdf5 latent files found under: {self.latent_dir}")

        self.meta = self._load_meta()
        self._validate_first_file()
        print(f"LatentDataset: found {len(self.files)} samples in {self.latent_dir}")

    def _discover_files(self, scene_list_file: Optional[str]) -> List[str]:
        if scene_list_file:
            scene_path = scene_list_file
            if not os.path.isabs(scene_path):
                scene_path = os.path.join(self.latent_dir, scene_path)
            if not os.path.isfile(scene_path):
                raise FileNotFoundError(f"scene_list_file not found: {scene_path}")
            with open(scene_path, "r", encoding="utf-8") as f:
                scenes = [line.strip() for line in f if line.strip()]
            files: List[str] = []
            for scene in scenes:
                scene_dir = os.path.join(self.latent_dir, scene)
                files.extend(glob(os.path.join(scene_dir, "**", "*.h5"), recursive=True))
                files.extend(glob(os.path.join(scene_dir, "**", "*.hdf5"), recursive=True))
            return sorted(set(files))

        return sorted(
            set(glob(os.path.join(self.latent_dir, "**", "*.h5"), recursive=True))
            | set(glob(os.path.join(self.latent_dir, "**", "*.hdf5"), recursive=True))
        )

    def _load_meta(self) -> Dict:
        meta_path = os.path.join(self.latent_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _validate_first_file(self) -> None:
        required = ["latent_image"] + [f"latent_{name}" for name in AOV_NAMES]
        with h5py.File(self.files[0], "r") as h5:
            missing = [key for key in required if key not in h5]
            if self.include_patch_tokens and "patch_tokens" not in h5:
                missing.append("patch_tokens")
            if self.require_prompt_embeds:
                for key in ("encoder_hidden_states", "pooled_projection"):
                    if key not in h5:
                        missing.append(key)
            if missing:
                raise KeyError(
                    f"Latent file {self.files[0]} is missing keys: {missing}. "
                    "Regenerate the cache with the corresponding prepare script flags."
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.files[index]
        item: Dict[str, torch.Tensor | str] = {"path": path}
        with h5py.File(path, "r") as h5:
            for key in ["latent_image"] + [f"latent_{name}" for name in AOV_NAMES]:
                item[key] = torch.from_numpy(h5[key][...]).float()

            if self.include_patch_tokens and "patch_tokens" in h5:
                item["patch_tokens"] = torch.from_numpy(h5["patch_tokens"][...]).float()

            if self.include_prompt_embeds:
                if "encoder_hidden_states" in h5 and "pooled_projection" in h5:
                    item["encoder_hidden_states"] = torch.from_numpy(h5["encoder_hidden_states"][...]).float()
                    item["pooled_projection"] = torch.from_numpy(h5["pooled_projection"][...]).float()
                elif self.require_prompt_embeds:
                    raise KeyError(f"Prompt embeddings missing in {path}")

            for key in ("roughness_flag", "metallic_flag", "irradiance_flag"):
                if key in h5:
                    item[key] = torch.as_tensor(h5[key][...]).long().reshape(())
                else:
                    item[key] = torch.tensor(1, dtype=torch.long)

            if "prompt" in h5.attrs:
                item["prompt"] = h5.attrs["prompt"]
            elif "prompt" in h5:
                raw = h5["prompt"][()]
                item["prompt"] = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            else:
                item["prompt"] = ""

        return item


def split_dataset(dataset: Dataset, train_ratio: float = 0.95, seed: Optional[int] = None):
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    if train_ratio == 1.0:
        return dataset, None
    train_size = int(train_ratio * len(dataset))
    train_size = min(max(train_size, 1), len(dataset) - 1)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    print(f"Splitting latent dataset: {train_size} train / {val_size} val")
    return random_split(dataset, [train_size, val_size], generator=generator)
