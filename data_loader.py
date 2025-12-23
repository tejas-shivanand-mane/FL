import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class ClientImageDataset(Dataset):
    def __init__(self, client_id, transform=None, root="data"):
        self.client_dir = os.path.join(root, f"client_{client_id}")
        self.img_dir = os.path.join(self.client_dir, "train")
        self.transform = transform

        self.samples = []
        with open(os.path.join(self.client_dir, "labels.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("L")  # MNIST grayscale

        if self.transform:
            img = self.transform(img)

        return img, label
