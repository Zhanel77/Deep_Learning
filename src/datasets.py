import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ATTRS = ["Smiling", "Eyeglasses", "Male", "Young"]  # можно поменять

def get_celeba_dataloaders(
    data_dir="./data",
    img_size=64,
    batch_size=128,
    num_workers=2,
    download=True,
):
    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # -> [-1, 1]
    ])

    train_ds = datasets.CelebA(
        root=data_dir,
        split="train",
        target_type="attr",
        transform=tfm,
        download=download
    )
    val_ds = datasets.CelebA(
        root=data_dir,
        split="valid",
        target_type="attr",
        transform=tfm,
        download=download
    )

    # индексы нужных атрибутов
    attr_names = train_ds.attr_names
    attr_idx = [attr_names.index(a) for a in ATTRS]

    def collate_fn(batch):
        imgs, attrs = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        attrs = torch.stack(attrs, dim=0)  # (B, 40), значения -1/1
        attrs = attrs[:, attr_idx]
        attrs = (attrs > 0).float()        # -> 0/1
        return imgs, attrs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader
