import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from models.model import *
from utils import *

transform = A.Compose([
    A.RandomCrop(32,32),
    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=127),
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet18()
trainset = Cifar10(root='./data', train=True,download=True, transform=transform)
testset = Cifar10(root='./data', train=False,download=True, transform=transform)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_dataloader(batch):
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,batch_size = batch, shuffle=True)
    return trainloader, testloader

def init_modeloptim(lr=0.01):
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
  return optimizer

def train(epochs, optimizer, trainloader):
  for epoch in range(epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      correct = 0
      for i, data in enumerate(trainloader, 0):
          # get the inputs
          inputs, labels = data

          inputs=inputs.to(device)
          labels=labels.to(device)
                          
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(labels.view_as(pred)).sum().item()
          acc = 100. * correct / len(trainloader.dataset)
          if i % 2000 == 1999:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
      print("Accuracy is:",acc)

  print('Finished Training')

def test():
  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images=images.to(device)
          labels=labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
