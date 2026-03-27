import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_dino_feature import extract_patch_tokens_min_windows

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm([channels])
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x

class IMAA(nn.Module):
    def __init__(self, 
                 dino_model=None, 
                 processor=None,
                 num_maps=5, 
                 map_embedding_dim=256, 
                 common_dim=128, 
                 conv_channels=[128,64],
                 dino_patch_dim=768):
        super().__init__()
        self.dino = dino_model
        self.processor = processor
        if self.dino is not None:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False

        self.num_maps = num_maps
        self.map_embedding_dim = map_embedding_dim
        self.common_dim = common_dim
        self.dino_patch_dim = dino_patch_dim

        # learnable map embeddings
        self.map_embedding = nn.Parameter(torch.randn(num_maps, map_embedding_dim))

        self.dino_proj = nn.Conv2d(dino_patch_dim, common_dim, kernel_size=1)
        self.map_proj = nn.Linear(map_embedding_dim, common_dim)

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(common_dim * 2, common_dim, 1),
            LayerNorm2d(common_dim),
            nn.ReLU(),
            nn.Conv2d(common_dim, common_dim, 3, padding=1)
        )

        conv_layers = []
        in_ch = common_dim
        for out_ch in conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        conv_layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))  # single channel mask
        self.conv_head = nn.Sequential(*conv_layers)


    def forward(self, image=None, patch_tokens=None, output_size=None, map_ids=None):
        """
        image: (B, 3, H, W)
        patch_tokens: (B, H_d, W_d, dino_patch_dim)
        output_size: tuple (H_out, W_out)
        return: (B, num_maps, H_out*W_out)
        """
        if patch_tokens is None:
            assert self.dino is not None and image is not None, "need dino_model or patch_tokens"
            patch_tokens = extract_patch_tokens_min_windows(image, self.dino, self.processor,
                                                            window_size=224, device=image.device)

        B = patch_tokens.size(0)

        # (B, C, H, W)
        dino_feat_map = patch_tokens.permute(0, 3, 1, 2)
        dino_proj = self.dino_proj(dino_feat_map)  # (B, common_dim, H, W)
        # print("dino_proj.shape:", dino_proj.shape)  # (B, common_dim, H, W)

        # expand map embeddings
        map_emb = self.map_embedding[map_ids] # (B, map_embedding_dim)
        # print("map_emb.shape:", map_emb.shape) 
        map_proj = self.map_proj(map_emb)  # (B, common_dim)
        # print("map_proj.shape:", map_proj.shape)  


        map_proj = map_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dino_proj.size(2), dino_proj.size(3))
        # print("map_proj.shape after expand:", map_proj.shape)
        # (B, common_dim, H, W)


        # element-wise product → (B, common_dim, H, W)
        fused_map = self.fusion_layer(
            torch.cat([dino_proj, map_proj.expand_as(dino_proj)], dim=1)
        )

        raw_gating_map = self.conv_head(fused_map)  # (B, 1, H, W)
        # print("raw_gating_map.shape:", raw_gating_map.shape)

        # up sample
        if output_size is not None:
            aligned_map = F.interpolate(raw_gating_map, size=output_size, mode='bilinear', align_corners=False)
        else:
            aligned_map = raw_gating_map

        w_gating = torch.sigmoid(aligned_map)

        return w_gating

def build_attn_mask(
    w_gating,              # [B, 1, H, W] or [B, img_len]
    text_len, img_len,     # ints
    lam,                   # float, scaling factor lambda
):
    B = w_gating.shape[0]
    Tk = text_len + img_len


    if w_gating.dim() == 4:
        w_gating = w_gating.view(B, -1)  # [B, 1, H, W] -> [B, H*W]
    
    g = lam * w_gating                  
    
    actual_img_len = g.shape[1]
    if actual_img_len != img_len:
        print(f"Warning: actual_img_len ({actual_img_len}) != expected img_len ({img_len})")

        if actual_img_len > img_len:
            g = g[:, :img_len]
        else:

            padding = torch.zeros(B, img_len - actual_img_len, device=g.device, dtype=g.dtype)
            g = torch.cat([g, padding], dim=1)

    col_bias = torch.zeros(B, Tk, device=w_gating.device, dtype=w_gating.dtype)
    col_bias[:, text_len:] = g             

    col_bias = col_bias.view(B, 1, 1, Tk)

    attn_mask = col_bias

    return attn_mask

