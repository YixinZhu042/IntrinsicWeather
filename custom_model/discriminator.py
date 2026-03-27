import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim

class PatchDiscriminator(nn.Module):
    """
    A simple PatchGAN-style discriminator.
    Input: [B, C, H, W] (e.g. RGB image)
    Output: [B, 1, H/16, W/16] (patch-level real/fake prediction)
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_channels, base_channels, norm=False),   # [B, 64, H/2, W/2]
            block(base_channels, base_channels*2),           # [B, 128, H/4, W/4]
            block(base_channels*2, base_channels*4),         # [B, 256, H/8, W/8]
            block(base_channels*4, base_channels*8),         # [B, 512, H/16, W/16]
            nn.Conv2d(base_channels*8, 1, 4, stride=1, padding=1)  # Patch real/fake
        )

    def forward(self, x):
        return self.net(x)

