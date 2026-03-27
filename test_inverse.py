import os
import argparse
import csv
import random
import cv2
import torch
import torchvision
from PIL import Image
import numpy as np
from inverse_renderer_pipeline import StableDiffusion3InstructPix2PixPipeline
from diffusers.models.autoencoders import AutoencoderKL
from custom_model.transformer import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from data.WeatherSynthetic import WeatherSynthetic
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms
from IMAA import IMAA, build_attn_mask
from extract_dino_feature import extract_patch_tokens_min_windows
from transformers import AutoImageProcessor, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model prediction will be saved.")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu', 'cuda:0', 'cuda:1').")
parser.add_argument("--save_images", action="store_true", default=True, help="Whether to save output images (default: True).")
parser.add_argument("--save_results", action="store_true", default=True, help="Whether to save evaluation results to CSV (default: True).")
parser.add_argument("--no_save_images", action="store_true", help="Disable saving images (overrides --save_images).")
parser.add_argument("--no_save_results", action="store_true", help="Disable saving results (overrides --save_results).")
args = parser.parse_args()

save_images = args.save_images and not args.no_save_images
save_results = args.save_results and not args.no_save_results


device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
print(f"Using device: {device}")
print(f"Save images: {save_images}")
print(f"Save results: {save_results}")

if (save_images or save_results) and not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 42
set_seed(seed)

dataset_3 = WeatherSynthetic(
    "WeatherSynthetic_dataset",
    scene_list_file="scene.txt",
    imWidth=512,
    imHeight=512,
)

train_ratio = 0.95
test_ratio = 1 - train_ratio
total_size = len(dataset_3)
train_size = int(train_ratio * total_size)
test_size = total_size - train_size

print(f"Splitting dataset: {train_size} for training, {test_size} for testing.")
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset_3, [train_size, test_size], generator=generator)


dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    

transformer = SD3Transformer2DModel.from_pretrained(
        "sd3.5_medium", 
        subfolder="transformer",
        low_cpu_mem_usage=False,
        device_map=None
    )
in_channels = 32
out_channels = transformer.pos_embed.proj.out_channels
transformer.register_to_config(in_channels=in_channels)

with torch.no_grad():
    new_proj = torch.nn.Conv2d(
        in_channels, out_channels,transformer.pos_embed.proj.kernel_size, transformer.pos_embed.proj.stride, transformer.pos_embed.proj.padding
    )
    new_proj.weight.zero_()
    new_proj.weight[:, :16, :, :].copy_(transformer.pos_embed.proj.weight)
    transformer.pos_embed.proj = new_proj

state_dict0 = torch.load("checkpoints/InverseRenderer-1024/pytorch_model-00001-of-00002.bin", map_location="cpu", weights_only=True)
state_dict1 = torch.load("checkpoints/InverseRenderer-1024/pytorch_model-00002-of-00002.bin", map_location="cpu", weights_only=True)


state_dict = {**state_dict0, **state_dict1}

imaa_dict = torch.load("checkpoints/InverseRenderer-1024/imaa.pth", weights_only=True)


transformer.load_state_dict(state_dict)
transformer.eval()

imaa = IMAA(dino_model=None, processor=None, num_maps=5, map_embedding_dim=256, common_dim=128).to(device)
imaa.load_state_dict(imaa_dict["model_state_dict"])
imaa.eval()  

pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
                       "sd3.5_medium", 
                        transformer=transformer,
                        use_safetensors=True
                    ).to(device, dtype=torch.float16)

lpips_model = lpips.LPIPS(net="vgg")
csv_path = os.path.join(args.output_dir, "psnr.csv") if save_results else None

def write_csv(data_row):
    if save_results and csv_path:
        path = csv_path
        with open(path, mode='a', newline='', encoding='utf-8_sig') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)

processor = AutoImageProcessor.from_pretrained('dino')
model = AutoModel.from_pretrained('dino').to(device)


if save_results and csv_path:
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["step", "albedo-psnr", "albedo-ssim", "albedo-lpips", "normal-psnr", "normal-ssim", "normal-lpips", "roughness-psnr", "roughness-ssim", "roughness-lpips", 
         "metallic-psnr", "metallic-ssim", "metallic-lpips", "irradaince-psnr", "irradaince-ssim",  "irradaince-lpips"])

for step, batch in enumerate(dataloader):
    original_image = batch["im"].to(device, dtype=torch.float16)
    patch_tokens = extract_patch_tokens_min_windows(original_image, model, processor,
                                                    window_size=224, device=original_image.device)

    # print(original_image.min(), original_image.max()
    if save_images:
        image = (original_image[0] / 2 + 0.5) * 255
        image = image.permute(1,2,0).cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(args.output_dir, f"image_{step}.png"))

    width, height = original_image.shape[1], original_image.shape[2]
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts_list = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (lighting)"
    }
    record = []
    record.append(step)
    print(original_image.shape)
    
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

        output_size = (original_image.shape[2] // 16, original_image.shape[3] // 16)
        img_len = output_size[0] * output_size[1]
        map_aware_mask = imaa(patch_tokens=patch_tokens, output_size=output_size, map_ids=torch.tensor([i]).to(device))
        attn_mask = build_attn_mask(map_aware_mask, 154, img_len, 0.7)

        output_maps = pipe(
            image=original_image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=0,
            image_guidance_scale=0,
            num_inference_steps=50,
            strength=1.0,
            output_type="np",
            aov=[aov],
            map_aware_mask=attn_mask.to(original_image.device),
        )[0][0]
        

        gt = batch[aov][0].permute(1,2,0).cpu().numpy()
        output_aov = output_maps
        print(type(output_aov))
        print(output_aov.shape)

      
        gt = (gt / 2 + 0.5).clip(0, 1)
        if aov == "albedo" or aov == "irradiance":
            gt = gt ** (1.0 / 2.2)
        gt = gt * 255
        if save_images:
            gt_image = gt.astype(np.uint8)
            gt_image = Image.fromarray(gt_image)
            gt_image.save(os.path.join(args.output_dir, f"image_{step}_gt_{aov}.png"))
     
        output_aov *= 255
 
        psnr_wo_tag = psnr(gt, output_aov, data_range=255)
        ssim_metric = ssim(gt, output_aov, channel_axis=2, data_range=255)

        output_aov_normalized = output_aov / 255.0  # to [0, 1]
        output_aov_tensor = torch.tensor(output_aov_normalized).permute(2, 0, 1).float()
        output_aov_tensor = output_aov_tensor * 2 - 1  # to [-1, 1]
        lpips_metric = lpips_model(batch[aov][0].cpu(), output_aov_tensor.cpu())
        lpips_metric = np.round(lpips_metric.detach().numpy().item(), decimals=4)
        psnr_wo_tag = np.round(psnr_wo_tag, 4)
        ssim_metric = np.round(ssim_metric, 4)
        record.append(float(psnr_wo_tag))
        record.append(float(ssim_metric))
        # print(type(lpips_metric))
        record.append(float(lpips_metric))
        print(aov, psnr_wo_tag, ssim_metric, lpips_metric)
        # plt.imshow(img_wo_tag)
        # plt.show()
        if save_images:
            save_aov = Image.fromarray(output_aov.astype(np.uint8))
            save_path = os.path.join(args.output_dir, f"image_{step}_{aov}.png")
            save_aov.save(save_path)
    print(record)
    write_csv(record)


