import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

# flatten the images
new_train_dataset = []

for img, label in train_dataset:
    img = img.flatten()
    new_train_dataset.append((img, label))
    
new_test_dataset = []

for img, label in test_dataset:
    img = img.flatten()
    new_test_dataset.append((img, label))

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=new_train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=new_test_dataset, batch_size=64, shuffle=True)

# Define the model
# Logistic regression model
class LogReg(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogReg, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)

        
    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        
        return out
    
# Neural network model
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # First hidden layer
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        # Second hidden layer
        self.linear2 = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out = self.linear2(out)
        
        return out
    
# Define constants and Create the model
SIZE_IMG = len(new_train_dataset[0][0])
NUM_OUTCOMES = 10 # 10 digits as possible outcomes (0,1,2,3,4,5,6,7,8,9)
model = LogReg(SIZE_IMG, NUM_OUTCOMES).to(device)

# Define the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get the data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Get the correct shape
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = loss_function(scores, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
        
        print(loss.item())

