import os
import csv
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image

NUM_CLIENTS = 2
SAMPLES_PER_CLIENT = [500, 1500]  # customize per client
DATA_ROOT = "data"

os.makedirs(DATA_ROOT, exist_ok=True)

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)


indices = np.arange(len(dataset))
np.random.seed(42)
np.random.shuffle(indices)

splits = []
start = 0
for n in SAMPLES_PER_CLIENT:
    splits.append(indices[start:start+n])
    start += n

for client_id, client_indices in enumerate(splits):
    client_dir = os.path.join(DATA_ROOT, f"client_{client_id}")
    img_dir = os.path.join(client_dir, "train")
    os.makedirs(img_dir, exist_ok=True)

    labels_path = os.path.join(client_dir, "labels.csv")

    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for i, idx in enumerate(client_indices):
            img, label = dataset[idx]
            img_path = os.path.join(img_dir, f"{i}.png")

            Image.fromarray((img.squeeze().numpy() * 255).astype("uint8")).save(img_path)
            writer.writerow([f"{i}.png", label])

    print(f"Client {client_id}: {len(client_indices)} samples written")
