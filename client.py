import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import flwr as fl
from torch.utils.data import DataLoader, Subset

# -----------------------------
# Client identity
# -----------------------------
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
NUM_CLIENTS = 2   # total number of clients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# Data partitioning (IID)
# -----------------------------
def load_data(client_id, num_clients):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    indices = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    client_indices = splits[client_id]

    trainloader = DataLoader(
        Subset(trainset, client_indices),
        batch_size=32,
        shuffle=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
    )

    return trainloader, testloader

# -----------------------------
# Flower client
# -----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data(CLIENT_ID, NUM_CLIENTS)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for x, y in self.trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples

        return (
            self.get_parameters(config),
            total_samples,
            {"train_loss": avg_loss},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                loss_sum += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += x.size(0)

        avg_loss = loss_sum / total
        accuracy = correct / total

        return (
            avg_loss,
            total,
            {"accuracy": accuracy},
        )

# -----------------------------
# Start client
# -----------------------------
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=FlowerClient(),
    )
