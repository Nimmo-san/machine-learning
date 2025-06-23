import torch # core library
import torch.nn as nn # neural network layers and loss functions
import torch.optim as optim # optimisers such as SGD, Adam, etc

from torchvision import datasets, transforms # datasets for common datasets, transforms for preprocessing functionality
from torch.utils.data import DataLoader # efficiently loads and shuffles the dataset in batches
from torch.utils.data import random_split


# Defining the cnn model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), # conv2d extracts patterns from 2d images 
            nn.ReLU(),              # relu activation function that introduced non-linearty
            nn.MaxPool2d(2),        # maxpool2d downsamples spatial dimensions (reduces size, increase abstraction)
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),           # flatten turns a 2d tensor into a 1d vector for feeding into dense layers
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 output classes for digits 0-9, fully connected layer for classification
        )

    def forward(self, x):
        return self.model(x)



# load and preprocess the dataset
transform = transforms.Compose([
    transforms.ToTensor(), # converts pil image (0-255) to tensor [0.0, 1.0], scales the values accordingly
    transforms.Normalize((.1307,), (.3081,)) # normalises the data mean and std from mnist to improve training stability
])

# full MNIST training dataset
# full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# splitting 48,000 for training, 12,000 for validation
# train_size = int(0.8 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# dataLoaders after splitting data for batch processing
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)

# without splitting data
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # batches data and shuffles it every epoch


# # model, loss, optimiser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device=device)
loss_fn = nn.CrossEntropyLoss() # combines softmax and negative log-likelihood, good for classification
optimiser = optim.Adam(model.parameters(), lr=0.001) # efficient optimiser with adaptive learning rates


# # training loop
for epoch in range(1, 6):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
    print(f"{epoch=}, Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "mnist_cnn.pt")