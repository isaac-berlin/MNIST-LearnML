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
from sklearn.model_selection import KFold

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

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# Define the model Class
# create a CNN model
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
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
    

folds = 5
fold_size = len(train_dataset) // folds

# define the loss
loss_func = nn.CrossEntropyLoss()

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

kFold = KFold(n_splits=folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kFold.split(full_dataset)):
    print(f'Fold {fold + 1}')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    train_loader = torch.utils.data.DataLoader(dataset=full_dataset, batch_size=64, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(dataset=full_dataset, batch_size=64, sampler=test_subsampler)
    
    model = CNN(NUM_OUTCOMES).to(device)
    model.apply(reset_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)
    
    # Evaluate the model
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}')
    print(f'Fold {fold + 1} done')