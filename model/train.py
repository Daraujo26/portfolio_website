from einops.layers.torch import Rearrange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch.optim as optim
from torchvision.transforms import RandomRotation, Compose, ToTensor, Normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_train = torchvision.datasets.MNIST(".", train=False, download=True, transform=torchvision.transforms.Compose([
    RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307), (0.3081)),
]))
dataset_test = torchvision.datasets.MNIST(".", train=False, download=True, transform=torchvision.transforms.Compose([
    RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307), (0.3081)),
]))

dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=256)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

model = CNN().to(device)

print(f"{sum([np.prod(x.data.shape) for x in model.parameters()]):,}")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

# ... [Training loop remains mostly the same but with the following modification for evaluation]

for epoch in range(35):
    print(f"Starting epoch {epoch}...")
    correct = 0; total = 0
    pbar = tqdm(dataloader_train)
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()     
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        pbar.set_description(f"loss: {loss:.4f}")
        loss.backward()
        optimizer.step()
        correct += (y_hat.argmax(dim=1) == y).sum()
        total += y.shape[0]
    print(f"Training Acc: {correct/total*100:.4f}%")

    correct = 0
    total = 0
    for x, y in dataloader_test:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for evaluation
            x = x.to(device); y = y.to(device)
            y_hat = model(x)
            y_hat = y_hat.argmax(dim=1)
            correct += (y_hat == y).sum()
            total += y.shape[0]
    print(f"Acc: {correct/total*100:.4f}%")
    
    
torch.save(model.state_dict(), 'proj_model.pt')

