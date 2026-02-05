import os
import argparse
import torch
from torch import optim
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.models.vae import CVAE, vae_loss
from src.datasets_kaggle import get_kaggle_celeba_loaders, ATTRS


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_anneal_beta(epoch, warmup_epochs=10, max_beta=1.0):
    # линейный разогрев beta от 0 до max_beta
    if epoch >= warmup_epochs:
        return max_beta
    return max_beta * (epoch / warmup_epochs)


@torch.no_grad()
def save_reconstructions(model, batch, out_path, n=16):
    model.eval()
    x, a = batch
    x = x[:n]
    a = a[:n]
    x_hat, _, _ = model(x, a)

    # grid: верх - real, низ - recon
    grid = torch.cat([x, x_hat], dim=0)
    grid = make_grid(grid, nrow=n, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


@torch.no_grad()
def save_samples(model, attrs_fixed, out_path, n=16, latent_dim=128, device="cpu"):
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    x_gen = model.decode(z, attrs_fixed.to(device))
    grid = make_grid(x_gen, nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)  # windows safe
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./runs/cvae")
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

    model = CVAE(
        img_channels=3,
        img_size=args.img_size,
        attr_dim=len(ATTRS),
        latent_dim=args.latent_dim
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = kl_anneal_beta(epoch-1, warmup_epochs=args.warmup_epochs, max_beta=args.max_beta)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} beta={beta:.3f}")
        total_loss = total_recon = total_kl = 0.0
        n_steps = 0

        for x, a in pbar:
            x = x.to(device)
            a = a.to(device)

            x_hat, mu, logvar = model(x, a)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_steps += 1

            pbar.set_postfix({
                "loss": total_loss / n_steps,
                "recon": total_recon / n_steps,
                "kl": total_kl / n_steps
            })

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            vloss = vrecon = vkl = 0.0
            vsteps = 0
            for x, a in val_loader:
                x = x.to(device)
                a = a.to(device)
                x_hat, mu, logvar = model(x, a)
                loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)
                vloss += loss.item()
                vrecon += recon.item()
                vkl += kl.item()
                vsteps += 1

            vloss /= max(vsteps, 1)
            vrecon /= max(vsteps, 1)
            vkl /= max(vsteps, 1)

        print(f"[VAL] loss={vloss:.4f} recon={vrecon:.4f} kl={vkl:.4f}")

        # save reconstructions
        first_batch = next(iter(val_loader))
        xb, ab = first_batch
        xb = xb.to(device)
        ab = ab.to(device)
        recon_path = os.path.join(args.out_dir, "images", f"recon_epoch{epoch}.png")
        save_reconstructions(model, (xb, ab), recon_path, n=min(16, xb.size(0)))

        # save samples with fixed attrs (take first 16 attrs from val batch)
        attrs_fixed = ab[:16]
        samp_path = os.path.join(args.out_dir, "images", f"samples_epoch{epoch}.png")
        save_samples(model, attrs_fixed, samp_path, n=16, latent_dim=args.latent_dim, device=device)

        # save best weights
        if vloss < best_val:
            best_val = vloss
            wpath = os.path.join(args.out_dir, "weights", "cvae_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "attr_names": ATTRS,
                "latent_dim": args.latent_dim
            }, wpath)
            print("✅ Saved best:", wpath)

    print("Done.")


if __name__ == "__main__":
    main()
