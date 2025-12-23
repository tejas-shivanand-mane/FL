import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Same model as FL
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
# Data (full dataset)
# -----------------------------
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

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# -----------------------------
# Training
# -----------------------------
model = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

NUM_EPOCHS = 20  # ← MUST match FL rounds

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in trainloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    avg_loss = running_loss / len(trainset)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += x.size(0)

accuracy = correct / total
print(f"Centralized accuracy: {accuracy:.4f}")
