import torch # core library
import torch.nn as nn # neural network layers and loss functions
import torch.optim as optim # optimisers such as SGD, Adam, etc

from torchvision import datasets, transforms # datasets for common datasets, transforms for preprocessing functionality
from torch.utils.data import DataLoader  # efficiently loads and shuffles the dataset in batches
# from torch.utils.data import random_split
from pathlib import Path

data_dir = "./data"
model_save_path = Path("mnist_cnn.pt")


# Defining the cnn model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X



# load and preprocess the dataset
transform = transforms.Compose([
    transforms.ToTensor(), # converts pil image (0-255) to tensor [0.0, 1.0], scales the values accordingly
    transforms.Normalize((.1307,), (.3081,)) # normalises the data mean and std from mnist to improve training stability
])

# without splitting data
train_dataset = datasets.MNIST(
    root=data_dir, train=True, transform=transform, download=True
)
val_dataset = datasets.MNIST(
    root=data_dir, train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # batches data and shuffles it every epoch
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

# # model, loss, optimiser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device=device)
loss_fn = nn.CrossEntropyLoss() # combines softmax and negative log-likelihood, good for classification
optimiser = optim.Adam(model.parameters(), lr=1e-3) # efficient optimiser with adaptive learning rates



EPOCH = 50
best_acc = 0.0

# # training loop
for epoch in range(1, EPOCH+1):
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()

    # validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"{epoch=}: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
    # if best_acc >= .98:
    #     break

print(f"{best_acc:.4f}")
model.cpu()
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path.resolve()}")