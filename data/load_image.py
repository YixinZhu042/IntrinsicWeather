import cv2
import torch
import numpy as np
from torchvision.transforms import Resize, InterpolationMode


def convert_rgb_2_XYZ(rgb):
    # Reference: https://web.archive.org/web/20191027010220/http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # rgb: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(rgb)
    XYZ[:, :, 0] = (
        0.4124564 * rgb[:, :, 0] + 0.3575761 * rgb[:, :, 1] + 0.1804375 * rgb[:, :, 2]
    )
    XYZ[:, :, 1] = (
        0.2126729 * rgb[:, :, 0] + 0.7151522 * rgb[:, :, 1] + 0.0721750 * rgb[:, :, 2]
    )
    XYZ[:, :, 2] = (
        0.0193339 * rgb[:, :, 0] + 0.1191920 * rgb[:, :, 1] + 0.9503041 * rgb[:, :, 2]
    )
    return XYZ


def convert_XYZ_2_Yxy(XYZ):
    # XYZ: (h, w, 3)
    # Yxy: (h, w, 3)
    Yxy = torch.ones_like(XYZ)
    Yxy[:, :, 0] = XYZ[:, :, 1]
    sum = torch.sum(XYZ, dim=2)
    inv_sum = 1.0 / torch.clamp(sum, min=1e-4)
    Yxy[:, :, 1] = XYZ[:, :, 0] * inv_sum
    Yxy[:, :, 2] = XYZ[:, :, 1] * inv_sum
    return Yxy


def convert_rgb_2_Yxy(rgb):
    # rgb: (h, w, 3)
    # Yxy: (h, w, 3)
    return convert_XYZ_2_Yxy(convert_rgb_2_XYZ(rgb))


def convert_XYZ_2_rgb(XYZ):
    # XYZ: (h, w, 3)
    # rgb: (h, w, 3)
    rgb = torch.ones_like(XYZ)
    rgb[:, :, 0] = (
        3.2404542 * XYZ[:, :, 0] - 1.5371385 * XYZ[:, :, 1] - 0.4985314 * XYZ[:, :, 2]
    )
    rgb[:, :, 1] = (
        -0.9692660 * XYZ[:, :, 0] + 1.8760108 * XYZ[:, :, 1] + 0.0415560 * XYZ[:, :, 2]
    )
    rgb[:, :, 2] = (
        0.0556434 * XYZ[:, :, 0] - 0.2040259 * XYZ[:, :, 1] + 1.0572252 * XYZ[:, :, 2]
    )
    return rgb


def convert_Yxy_2_XYZ(Yxy):
    # Yxy: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(Yxy)
    XYZ[:, :, 0] = Yxy[:, :, 1] / torch.clamp(Yxy[:, :, 2], min=1e-6) * Yxy[:, :, 0]
    XYZ[:, :, 1] = Yxy[:, :, 0]
    XYZ[:, :, 2] = (
        (1.0 - Yxy[:, :, 1] - Yxy[:, :, 2])
        / torch.clamp(Yxy[:, :, 2], min=1e-4)
        * Yxy[:, :, 0]
    )
    return XYZ


def convert_Yxy_2_rgb(Yxy):
    # Yxy: (h, w, 3)
    # rgb: (h, w, 3)
    return convert_XYZ_2_rgb(convert_Yxy_2_XYZ(Yxy))


def load_ldr_image(image_path, from_srgb=False, clamp=False, normalize=False):
    # Load png or jpg image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if from_srgb:
        # Convert from sRGB to linear RGB
        image = image**2.2
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)


def load_exr_raw(image_path, width, height):
    """加载原始 HDR EXR，不进行 tonemapping，用于整段视频一致 tonemapping 防闪烁"""
    image = cv2.imread(image_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.nan_to_num(image, nan=0.0, posinf=1, neginf=0.0)
    image = np.clip(image, 0, 1e4)  # 避免极端值
    image = torch.from_numpy(image.astype("float32"))
    image[~torch.isfinite(image)] = 0
    image = image.permute(2, 0, 1)
    resize_transform = Resize(size=(height, width), interpolation=InterpolationMode.BILINEAR)
    image = resize_transform(image)
    return image  # (C, H, W)


def load_exr_image(image_path, width, height, tonemaping=False, clamp=False, normalize=False, driving=False):
    image = cv2.imread(image_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # print(image_path[-15:-3], image.min(), image.max())
    image = np.nan_to_num(image, nan=0.0, posinf=1, neginf=0.0)

    if tonemaping:
        image = image.clip(0,999)
        tonemap = cv2.createTonemapReinhard(1.5, 0.5, 0.5, 0.5)
        image = tonemap.process(image)
    image = torch.from_numpy(image.astype("float32"))  # (h, w, c)
    image[~torch.isfinite(image)] = 0


    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)

    if normalize:
        # image = image * 2 - 1  # InteriorVerse 和 Hypersim 的normal本来就在[-1,1]
        image = image.clamp(-1,1)
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    elif driving:
        image = normalize_image(image, dim=-1, eps=1e-6) # DrivingScene的normal首先归一化
        image = 2.0 * image - 1.0 # DrivingScene的normal在[0,1]上
        image = normalize_image(image, dim=-1, eps=1e-6)
        # print("normal: ", image.min(), image.max())
        image = image.clamp(-1, 1)
        # image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
        # print("normal:", image[0])
    else: #其余属性直接放缩到[-1,1]上
        image = image * 2.0 - 1.0

    image = image.permute(2, 0, 1)  # returns (c, h, w)
    # print(image_path[-15:-3], image.min(), image.max(), image.norm(p=2))
    # Resize
    resize_transform = Resize(size=(height, width), interpolation=InterpolationMode.BILINEAR)
    image = resize_transform(image)
    # print(f"load after resize: {image.shape}")

    return image

def normalize_image(image: torch.Tensor, p: float=2.0, dim: int=-1, eps: float=1e-6):
    sky_mask = (image.sum(dim=-1)<=-2.9)
    # print(image.sum(dim=-1)[0])
    # print(sky_mask)
    # print(image.sum(dim=-1))
    denom = image.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(image).clone()
    denom[sky_mask] = 1
    image[sky_mask] = -1
    return image / denom