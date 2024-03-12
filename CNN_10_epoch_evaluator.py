import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ignite.metrics import ConfusionMatrix
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy


NUM_OUTCOMES = 10

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transforms for the data
t = transforms.Compose([
    transforms.ToTensor(), # convert the image to a pytorch tensor
])

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

# Load the saved model
model = CNN(28*28, NUM_OUTCOMES).to(device)
model.load_state_dict(torch.load('CNN_10_epoch.pth'))

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=t)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

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
        
    print(f'After 10 epochs, Got {n_correct}/{n_samples} with accuracy {float(n_correct)/float(n_samples)*100:.2f}')

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