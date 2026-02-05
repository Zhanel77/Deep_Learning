import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional VAE for 64x64 RGB faces.
    Conditioning: binary attribute vector (e.g., 4 attrs).
    Encoder: image + attrs -> (mu, logvar)
    Decoder: z + attrs -> reconstructed image
    """

    def __init__(self, img_channels=3, img_size=64, attr_dim=4, latent_dim=128, base_ch=64):
        super().__init__()
        assert img_size == 64, "This implementation assumes 64x64 images."

        self.img_channels = img_channels
        self.img_size = img_size
        self.attr_dim = attr_dim
        self.latent_dim = latent_dim

        # --- Encoder: conv stack -> feature vector ---
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, base_ch, 4, 2, 1),      # 64 -> 32
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),         # 32 -> 16
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),       # 16 -> 8
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),       # 8 -> 4
            nn.BatchNorm2d(base_ch*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_out_dim = base_ch * 8 * 4 * 4  # (B, base_ch*8, 4, 4)

        # We inject attributes via a small MLP and concatenate with encoder features
        self.attr_enc = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(self.enc_out_dim + 64, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim + 64, latent_dim)

        # --- Decoder: z + attrs -> feature map -> deconv stack ---
        self.attr_dec = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.dec_fc = nn.Linear(latent_dim + 64, self.enc_out_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),    # 16 -> 32
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch, img_channels, 4, 2, 1), # 32 -> 64
            nn.Tanh()  # output in [-1, 1]
        )

    def encode(self, x, attrs):
        h = self.enc(x).view(x.size(0), -1)
        a = self.attr_enc(attrs)
        h_cat = torch.cat([h, a], dim=1)
        mu = self.fc_mu(h_cat)
        logvar = self.fc_logvar(h_cat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, attrs):
        a = self.attr_dec(attrs)
        z_cat = torch.cat([z, a], dim=1)
        h = self.dec_fc(z_cat).view(z.size(0), -1, 4, 4)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x, attrs):
        mu, logvar = self.encode(x, attrs)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, attrs)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    """
    Reconstruction + beta * KL
    Use L1 loss for stable recon on faces.
    """
    recon = F.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl
