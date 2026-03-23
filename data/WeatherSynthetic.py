import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'True'

import os.path as osp
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from data.load_image import load_exr_image
import h5py
import json

class WeatherSynthetic(Dataset):
    def __init__(self, root_dir, scene_list_file, imWidth=512, imHeight=512, prompt_json_file=None):
        """
        root_dir: root dir, e.g., '/path/to/dataset'
        scene_list_file: a TXT file containing scene types 
        prompt_json_file: optional prompt file
        """
        self.root_dir = root_dir
        self.imWidth = imWidth
        self.imHeight = imHeight
        self.samples = []

        with open(os.path.join(root_dir, scene_list_file), 'r') as f:
            scenes = [line.strip() for line in f.readlines()]
        # print(scenes)

        self.prompt_dict = {}
        if prompt_json_file is not None:
            with open(os.path.join(root_dir, prompt_json_file), 'r') as pf:
                prompt_list = json.load(pf)
                self.prompt_dict = {item['image_path']: item['prompt'] for item in prompt_list}

        for scene in scenes:
            scene_path = os.path.join(root_dir, scene)

            weather_dir = os.path.join(scene_path, 'image')
            for weather in os.listdir(weather_dir):
                weather_path = os.path.join(weather_dir, weather)
                image_files = sorted([f for f in os.listdir(weather_path) if f.endswith('_image.exr')])
                for img_file in image_files:
                    base_id = img_file.split('_')[0]  
                    sample = {
                        'image': os.path.join(weather_path, f"{base_id}_image.exr"),
                        'irradiance': os.path.join(weather_path, f"{base_id}_irradiance.exr"),
                        'albedo': os.path.join(scene_path, 'property', 'albedo', f"{base_id}_albedo.exr"),
                        'normal': os.path.join(scene_path, 'property', 'normal', f"{base_id}_normal.exr"),
                        'roughness': os.path.join(scene_path, 'property', 'roughness', f"{base_id}_roughness.exr"),
                        'metallic': os.path.join(scene_path, 'property', 'metallic', f"{base_id}_metallic.exr"),
                    }
                    self.samples.append(sample)
        print(f"DrivingDataSet: Found {len(self.samples)} samples in {root_dir}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample["image"])
        im = load_exr_image(sample["image"], width=self.imWidth, height=self.imHeight, clamp=True, tonemaping=True)
        albedo = load_exr_image(sample["albedo"], width=self.imWidth, height=self.imHeight, clamp=True)
        normal = load_exr_image(sample["normal"], width=self.imWidth, height=self.imHeight, driving=True)
        metallic = load_exr_image(sample["metallic"], width=self.imWidth, height=self.imHeight, clamp=True)
        roughness = load_exr_image(sample["roughness"], width=self.imWidth, height=self.imHeight, clamp=True)
        irradiance = load_exr_image(sample["irradiance"], width=self.imWidth, height=self.imHeight, clamp=True, tonemaping=True)

        k = sample["image"]
        # Convert absolute image path -> "WeatherSynthetic/..." key
        # so it matches prompt json keys that are stored relative to the dataset root.
        k_rel = k
        
        try:
            root_prefix = os.path.normpath(self.root_dir)
            k_rel_inside = os.path.relpath(k, root_prefix)
            k_rel = os.path.join(os.path.basename(root_prefix), k_rel_inside)
        except Exception:
            # Fallback to original key if path conversion fails for any reason.
            k_rel = k
        print(k_rel)
        prompt = self.prompt_dict.get(k_rel, self.prompt_dict.get(k, ""))  


        batchDict = {
            'albedo': albedo,
            'normal': normal,
            'roughness': roughness,
            'metallic': metallic,
            'irradiance': irradiance,
            'im': im,
            'prompt': prompt,
        }

        return batchDict
    
if __name__ == "__main__":
    # Replace data_root with the actual path to your dataset folder
    data_root = "path/to/your/WeatherSynthetic" 
    train_dataset = WeatherSynthetic(data_root, "scene.txt", prompt_json_file="prompt.json")
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch["im"].shape)
        print(batch["albedo"].shape)
        print(batch["normal"].shape)
        print(batch["roughness"].shape)
        print(batch["metallic"].shape)
        print(batch["irradiance"].shape)
        # Save a visualization grid for the first sample in this batch.
        import matplotlib
        matplotlib.use("Agg")  # headless save
        import matplotlib.pyplot as plt

        def to_display(t):
            """Convert tensor (C,H,W) in [-1,1] to displayable numpy (H,W,3) or (H,W)."""
            t = t.detach().cpu()
            if t.dim() != 3:
                raise ValueError(f"Expected (C,H,W) tensor, got shape={tuple(t.shape)}")
            # Map [-1,1] -> [0,1] for visualization.
            t = (t + 1.0) / 2.0
            t = torch.clamp(t, 0.0, 1.0)
            c = t.shape[0]
            if c == 1:
                return t[0].numpy()
            # If more than 3 channels exist, just take the first 3.
            img = t[:3].permute(1, 2, 0).numpy()  # (H,W,3)
            return img

        # DataLoader default collate will give tensors for image-like keys.
        # For strings (prompt), it will typically give a list[str].
        prompt0 = ""
        if isinstance(batch.get("prompt", ""), (list, tuple)) and len(batch["prompt"]) > 0:
            prompt0 = batch["prompt"][0]

        keys_and_titles = [
            ("im", "image"),
            ("albedo", "albedo"),
            ("normal", "normal"),
            ("roughness", "roughness"),
            ("metallic", "metallic"),
            ("irradiance", "irradiance"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.reshape(-1)
        for ax, (k, title) in zip(axes, keys_and_titles):
            img = to_display(batch[k][0])  # first item in batch
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            elif title == "albedo" or title == "irradiance":
                ax.imshow(img ** (1.0 / 2.2))
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        fig.suptitle(f"{prompt0 if prompt0 else ''}", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(os.getcwd(), "weather_synthetic_vis.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved visualization to: {out_path}")
        break