import os
import argparse
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.models.gan import ConditionalGenerator, ConditionalCritic
from src.datasets_kaggle import get_kaggle_celeba_loaders, ATTRS


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_penalty(critic, real, fake, attrs, device):
    """
    WGAN-GP gradient penalty:
    GP = E[(||grad(D(x_hat))||2 - 1)^2]
    """
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    score = critic(x_hat, attrs)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad = grad.view(bsz, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


@torch.no_grad()
def save_samples(G, attrs_fixed, out_path, z_dim, device):
    G.eval()
    n = attrs_fixed.size(0)
    z = torch.randn(n, z_dim, device=device)
    fake = G(z, attrs_fixed.to(device))
    grid = make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_critic", type=int, default=5)      # critic updates per G update
    parser.add_argument("--lambda_gp", type=float, default=10.0) # GP weight
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./runs/cgan_wgangp")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "weights"), exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("ATTRS:", ATTRS)

    train_loader, val_loader = get_kaggle_celeba_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    G = ConditionalGenerator(z_dim=args.z_dim, attr_dim=len(ATTRS)).to(device)
    D = ConditionalCritic(attr_dim=len(ATTRS)).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # фиксируем атрибуты для sample-grid (берём из первого val batch)
    xb, ab = next(iter(val_loader))
    attrs_fixed = ab[:32]  # 32 сэмпла
    best_g = 1e9

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        d_loss_run = 0.0
        g_loss_run = 0.0
        steps = 0

        for real, attrs in pbar:
            real = real.to(device)
            attrs = attrs.to(device)
            bsz = real.size(0)

            # -------- Train Critic n_critic times --------
            for _ in range(args.n_critic):
                z = torch.randn(bsz, args.z_dim, device=device)
                fake = G(z, attrs).detach()

                d_real = D(real, attrs).mean()
                d_fake = D(fake, attrs).mean()

                gp = gradient_penalty(D, real, fake, attrs, device)
                d_loss = (d_fake - d_real) + args.lambda_gp * gp  # minimize

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # -------- Train Generator --------
            z = torch.randn(bsz, args.z_dim, device=device)
            fake = G(z, attrs)
            g_loss = -D(fake, attrs).mean()  # maximize critic score => minimize negative

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_loss_run += d_loss.item()
            g_loss_run += g_loss.item()
            steps += 1

            pbar.set_postfix({
                "d_loss": d_loss_run / steps,
                "g_loss": g_loss_run / steps
            })

        # save samples each epoch
        sample_path = os.path.join(args.out_dir, "images", f"samples_epoch{epoch}.png")
        save_samples(G, attrs_fixed, sample_path, args.z_dim, device)

        # save weights (simple: save last + best by g_loss avg)
        avg_g = g_loss_run / max(steps, 1)
        last_path = os.path.join(args.out_dir, "weights", "cgan_last.pt")
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "attr_names": ATTRS}, last_path)

        if avg_g < best_g:
            best_g = avg_g
            best_path = os.path.join(args.out_dir, "weights", "cgan_best.pt")
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "attr_names": ATTRS}, best_path)
            print("✅ Saved best:", best_path)

    print("Done.")


if __name__ == "__main__":
    main()
