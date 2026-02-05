import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

ATTRS = ["Smiling", "Eyeglasses", "Male", "Young"]

class CelebAKaggle(Dataset):
    def __init__(self, root_dir, split="train", img_size=64):
        """
        root_dir structure:
          root_dir/img_align_celeba/img_align_celeba/*.jpg  (или root_dir/img_align_celeba/*.jpg)
          root_dir/list_attr_celeba.csv or list_attr_celeba.txt
        """
        self.root_dir = root_dir
        self.split = split

        # Ищем папку с картинками (часто бывает вложенная)
        cand1 = os.path.join(root_dir, "img_align_celeba", "img_align_celeba")
        cand2 = os.path.join(root_dir, "img_align_celeba")
        self.img_dir = cand1 if os.path.isdir(cand1) else cand2

        # Ищем файл атрибутов
        csv_path = os.path.join(root_dir, "list_attr_celeba.csv")
        txt_path = os.path.join(root_dir, "list_attr_celeba.txt")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # ожидаем столбец image_id
            if "image_id" not in df.columns:
                # на всякий случай: первый столбец имя файла
                df.rename(columns={df.columns[0]: "image_id"}, inplace=True)
        elif os.path.exists(txt_path):
            # формат: первые 2 строки служебные
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            header = lines[1].split()
            rows = [l.split() for l in lines[2:]]
            df = pd.DataFrame(rows, columns=["image_id"] + header)
            for c in header:
                df[c] = df[c].astype(int)
        else:
            raise FileNotFoundError("Не найден list_attr_celeba.csv или list_attr_celeba.txt")

        # оставляем нужные колонки
        keep = ["image_id"] + ATTRS
        df = df[keep].copy()

        # приводим -1/1 -> 0/1
        for a in ATTRS:
            df[a] = (df[a].astype(int) > 0).astype(int)

        # Сплит: чтобы просто и быстро — делаем фиксированный разрез
        # (можно потом заменить на официальный partition файл, если захочешь)
        n = len(df)
        if split == "train":
            df = df.iloc[: int(0.9*n)]
        else:
            df = df.iloc[int(0.9*n):]

        self.df = df.reset_index(drop=True)

        self.tfm = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)
        attrs = torch.tensor(row[ATTRS].to_numpy(dtype="float32"), dtype=torch.float32)
        return img, attrs

def get_kaggle_celeba_loaders(data_dir="./data", img_size=64, batch_size=64, num_workers=2):
    train_ds = CelebAKaggle(data_dir, split="train", img_size=img_size)
    val_ds   = CelebAKaggle(data_dir, split="valid", img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
