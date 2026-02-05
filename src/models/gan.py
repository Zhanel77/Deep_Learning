import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: z + attrs -> image (3x64x64)
    DCGAN-style upsampling.
    """
    def __init__(self, z_dim=128, attr_dim=4, base_ch=64, img_ch=3):
        super().__init__()
        self.z_dim = z_dim
        self.attr_dim = attr_dim

        self.fc = nn.Linear(z_dim + attr_dim, base_ch * 8 * 4 * 4)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),    # 16->32
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1),       # 32->64
            nn.Tanh()  # [-1,1]
        )

    def forward(self, z, attrs):
        x = torch.cat([z, attrs], dim=1)
        h = self.fc(x).view(x.size(0), -1, 4, 4)
        return self.net(h)


class ConditionalCritic(nn.Module):
    """
    Critic (WGAN): image + attrs -> score
    Conditioning via attribute embedding tiled to spatial map and concatenated to image.
    """
    def __init__(self, attr_dim=4, base_ch=64, img_ch=3):
        super().__init__()
        self.attr_embed = nn.Linear(attr_dim, 64)

        # input channels: img_ch + 64 (attr map)
        self.net = nn.Sequential(
            nn.Conv2d(img_ch + 64, base_ch, 4, 2, 1),   # 64->32
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),     # 32->16
            nn.InstanceNorm2d(base_ch*2, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),   # 16->8
            nn.InstanceNorm2d(base_ch*4, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),   # 8->4
            nn.InstanceNorm2d(base_ch*8, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(base_ch*8*4*4, 1)

    def forward(self, x, attrs):
        # attrs -> (B,64) -> (B,64,1,1) -> tile to (B,64,64,64)
        a = self.attr_embed(attrs).unsqueeze(-1).unsqueeze(-1)
        a = a.expand(-1, -1, x.size(2), x.size(3))
        x_in = torch.cat([x, a], dim=1)
        h = self.net(x_in).view(x.size(0), -1)
        return self.fc(h)
