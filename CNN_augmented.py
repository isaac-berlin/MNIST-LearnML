import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from ignite.metrics import ConfusionMatrix
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
import os
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

NUM_OUTCOMES = 10

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms for the data

train_transform = transforms.Compose([
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

# TODO:: apply augmentations here
'''
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Random rotation by up to 15 degrees
    transforms.RandomErasing(),  # Random erasing
    transforms.GaussianBlur(kernel_size=5),  # Apply Gaussian blur
    transforms.ToTensor(), # convert the image to a pytorch tensor
])
'''

train_rotation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Random rotation by up to 15 degrees
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

train_gaussian_blur_transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=5),  # Apply Gaussian blur
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

train_random_erasing_transform = transforms.Compose([
    transforms.RandomErasing(),  # Random erasing
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

train_random_affine_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image horizontally and vertically by up to 10% of its size
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)

# Create a new dataset for augmented images
rotation_images = deepcopy(train_dataset)
rotation_images.transform = train_rotation_transform
blur_images = deepcopy(train_dataset)
blur_images.transform = train_gaussian_blur_transform
affine_images = deepcopy(train_dataset)
affine_images.transform = train_random_affine_transform
erase_images = deepcopy(train_dataset)
erase_images.transform = train_random_erasing_transform

# Concatenate the original dataset and the augmented dataset
train_dataset = torch.utils.data.ConcatDataset([train_dataset, rotation_images, blur_images, affine_images, erase_images])

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Define the model Class
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
num_epochs = 5

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

# Close TensorBoard writer
writer.close()

# Save the 5-epoch-trained model 
torch.save(model.state_dict(), 'CNN_5_epoch_augmented.pth')

# Traditional Accuracy evaluation
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
        
    print(f'After 5 epochs, Got {n_correct}/{n_samples} with accuracy {float(n_correct)/float(n_samples)*100:.2f}')

# Define a function to compute the confusion matrix
"""
def compute_confusion_matrix(model, test_loader):
    confusion_matrix = torch.zeros(NUM_OUTCOMES, NUM_OUTCOMES)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix
"""

# Define a function to compute confusion matrix
def compute_confusion_matrix(model, test_loader):
    confusion = ConfusionMatrix(num_classes=NUM_OUTCOMES)
    evaluator = create_supervised_evaluator(model, metrics={'confusion_matrix': confusion}, device=device)
    evaluator.run(test_loader)
    return confusion.compute().cpu().numpy()

# Compute confusion matrix
confusion_matrix = compute_confusion_matrix(model, test_loader)

# Print confusion matrix
class_labels = [str(i) for i in range(10)]
print("Confusion Matrix:")
print("True\Predicted\t" + "\t".join(class_labels))
for i, row in enumerate(confusion_matrix):
    print(f"{class_labels[i]}\t\t" + "\t".join(map(str, row)))