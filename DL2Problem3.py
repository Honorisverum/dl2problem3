# -*- coding: utf-8 -*-
"""DL2Problem3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(64 * 12 ** 2, 10)
        self.layers = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """
        :param x: torch.Size([bs, 1, 28, 28])
        """
        x = self.layers(x)
        x = self.fc(x.view(len(x), -1))
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
lr = 0.0001
epochs = 10
torch.cuda.manual_seed(1234)

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transform),
    batch_size=batch_size, shuffle=True
)

model = NeuralNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(model, device, dataloader, optimizer, epoch):
    model.train()
    loss_val = 0
    for batch in train_loader:
        input, target = [e.to(device) for e in batch]
        output = model(input)
        loss = F.nll_loss(output, target)
        loss_val += loss.item() * len(input)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Train {epoch} epoch |"
          f"loss: {loss_val / len(dataloader.dataset):.3f}")

def valid(model, device, dataloader, epoch):
    model.eval()
    loss_val, accuracy = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input, target = [e.to(device) for e in batch]
            output = model(input)
            loss_val += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    print(f"Test {epoch} epoch |"
          f"loss: {loss_val / len(dataloader.dataset):.3f} |"
          f"accuracy: {accuracy / len(dataloader.dataset):.3f}\n")

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    valid(model, device, test_loader, epoch)

def attack(image, epsilon, grad):
    attacked_image = image + epsilon * grad.sign()
    return attacked_image

def adversarial_test(model, device, dataloader, epsilon):
    accuracy = 0
    for batch in test_loader:
        input, target = [e.to(device) for e in batch]
        input.requires_grad = True
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        grad = input.grad.data
        attacked_data = attack(input, epsilon, input.grad.data)
        model.zero_grad()
        # re-classify
        output = model(attacked_data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()

    print(f"Epsilon: {epsilon:.2f} |"
          f"Accuracy: {accuracy / len(dataloader.dataset):.3f}")

"""## Atack"""

for epsilon in np.linspace(0., 0.5, 11):
    adversarial_test(model, device, test_loader, epsilon)

