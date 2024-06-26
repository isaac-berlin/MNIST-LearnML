import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os



# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms for the data
t = transforms.Compose([
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=t)

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=t)

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

SIZE_IMG = len(train_dataset[0][0])
NUM_OUTCOMES = 10 # 10 digits as possible outcomes (0,1,2,3,4,5,6,7,8,9)
      
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# TODO: Ask about stride and padding? - cutting off info on the edges?
# create a CNN model
class CNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = torch.nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
# Create the model
model = CNN(28*28, NUM_OUTCOMES).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set up TensorBoard writer
writer = SummaryWriter(os.path.join('runs', 'MNIST_experiment'))

# Train the model
n_total_steps = len(train_loader)
num_epochs = 11

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [64, 1, 28, 28]
        # resized: [64, 784]
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log loss to TensorBoard
        writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + i)
        
    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
        print(f'After {epoch} epochs, Got {n_correct}/{n_samples} with accuracy {float(n_correct)/float(n_samples)*100:.2f}')

# Close TensorBoard writer
writer.close()

# Save the 10-epoch-trained model 
torch.save(model.state_dict(), 'CNN_10_epoch.pth')
