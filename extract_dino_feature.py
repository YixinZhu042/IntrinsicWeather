from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from data.WeatherSynthetic import WeatherSynthetic
import numpy as np

def extract_patch_tokens_min_windows(images,
                                    model,
                                    processor,
                                    window_size=224,
                                    device="cuda:0"):
    """
    Tile the image with a minimal set of windows, extract DINO patch tokens (batched tensors).
    images: torch.Tensor (B, C, H, W)
    model: DINO vision transformer
    processor: AutoImageProcessor
    window_size: sliding window size (pixels)
    """
    B, C, H, W = images.shape
    C_out = model.config.hidden_size
    patch_size = model.config.patch_size

    token_avgs = []

    for b in range(B):
        img = images[b]
        # Tensor -> numpy
        if img.max() <= 1.0:
            img_np = (img.permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        else:
            img_np = img.permute(1,2,0).cpu().numpy().clip(0,255).astype(np.uint8)

        token_sum = torch.zeros((H//patch_size, W//patch_size, C_out), device=device)
        token_count = torch.zeros((H//patch_size, W//patch_size, 1), device=device)

        n_y = (H + window_size - 1) // window_size
        n_x = (W + window_size - 1) // window_size

        y_positions = [i*window_size for i in range(n_y-1)] + [H - window_size]
        x_positions = [i*window_size for i in range(n_x-1)] + [W - window_size]

        for y in y_positions:
            for x in x_positions:
                patch = img_np[y:y+window_size, x:x+window_size, :]
                inputs = processor(images=patch, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                patch_tokens = outputs.last_hidden_state[:, 1:, :]
                patch_tokens = patch_tokens.reshape(1, window_size//patch_size, window_size//patch_size, C_out).squeeze(0)

                token_sum[y//patch_size:y//patch_size+window_size//patch_size,
                          x//patch_size:x//patch_size+window_size//patch_size, :] += patch_tokens
                token_count[y//patch_size:y//patch_size+window_size//patch_size,
                            x//patch_size:x//patch_size+window_size//patch_size, 0] += 1

        token_avg = token_sum / token_count
        token_avgs.append(token_avg)

    return torch.stack(token_avgs, dim=0)  # (B, H//patch, W//patch, C_out)


if __name__ == "__main__":
    device = "cuda:0"
    # Set to your WeatherSynthetic root and DINO checkpoint (e.g. dinov2_base/).
    dataset = WeatherSynthetic("WeatherSynthetic_dataset", "scene.txt")
    image = dataset[0]["im"]
    print(image.shape)  # e.g. torch.Size([3, 512, 512])

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    patch_tokens = extract_patch_tokens_min_windows(image.unsqueeze(0), model, processor,
                                                    window_size=224, device=device)

    # e.g. torch.Size([1, 36, 36, 768]); align mask grid to latent grid as needed
    print(patch_tokens.shape)

# # Convert tensor to PIL Image
# if image.min() < 0:
#     image = image * 0.5 + 0.5
# if len(image.shape) == 4:  # If batch dimension exists
#     image = image.squeeze(0)
# if image.shape[0] == 3:  # If channels-first format
#     image = image.permute(1, 2, 0)
# image = (image * 255).clip(0, 255).to(torch.uint8).cpu().numpy()
# image = Image.fromarray(image)

# processor = AutoImageProcessor.from_pretrained('/data1/zyx/dino')
# model = AutoModel.from_pretrained('/data1/zyx/dino').to(device)

# inputs = processor(images=image, return_tensors="pt").to(device)
# print(inputs.pixel_values.shape) # torch.Size([1, 3, 224, 224])
# outputs = model(**inputs)
# print(outputs.keys())
# last_hidden_states = outputs.last_hidden_state
# patch_tokens = last_hidden_states[:, 1:]
# print(patch_tokens.shape)  # torch.Size([1, 256, 768])
