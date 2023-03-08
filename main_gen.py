import torchvision.transforms.functional as TF
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
from PIL import Image
import torchvision.transforms.functional as TF
from models.custom_resnet import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = A.Compose([
    A.RandomCrop(32,32),
    A.HorizontalFlip(p=0.5),
    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=127),
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

test_transform = A.Compose([
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

trainset = Cifar10(root='./data', train=True,download=True, transform=train_transform)
testset = Cifar10(root='./data', train=False,download=True, transform=test_transform)

def init_optim_gen(model, lr=0.01):
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  criterion = nn.CrossEntropyLoss()
  return optimizer, criterion

def init_optim_any(model, type, lr=0.01):
  if type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  else:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
  criterion = nn.CrossEntropyLoss()
  return optimizer, criterion

def init_dataloader(batch):
  trainloader = torch.utils.data.DataLoader(trainset,batch_size = 512, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset,batch_size = 512, shuffle=True)
  return trainloader, testloader

def train_gen(model, epochs, optimizer, trainloader, scheduler):
  loss_arr = []
  model.train()
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
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()

          # print statistics
          running_loss += loss.item()
          pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(labels.view_as(pred)).sum().item()
          acc = 100. * correct / len(trainloader.dataset)
          
          if i % 2000 == 1999:    # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
      print("Accuracy is:",acc)
      loss_arr.append(running_loss/len(trainloader.dataset))
      my_lr = scheduler.get_last_lr()
      print("LR is :",my_lr)
      print("Epoch is:",epoch)

  print('Finished Training')
  return loss_arr


def test_gen(model, testloader):
  inc = []
  pre = []
  correct = 0
  total = 0
  test_loss = 0
  model.eval()
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images=images.to(device)
          labels=labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          # Store wrongly predicted images
          if (predicted != labels).sum().item()>0:
            wrong_idx = ((predicted != labels).nonzero()[:,0])[0].item()
            wrong_samples = images[wrong_idx]
            wrong_preds = predicted[wrong_idx]
            actual_preds = labels.view_as(predicted)[wrong_idx]

            # Undo normalization
            wrong_samples = wrong_samples * 0.5
            wrong_samples = wrong_samples + 0.5
            wrong_samples = wrong_samples * 255.
            wrong_samples = wrong_samples.byte()
            img = TF.to_pil_image(wrong_samples)
            # print(img.shape)
            inc.append(img)
            pre.append(wrong_preds)

      plot_arr = []
      for i in range(len(inc)):
        # plot_arr.append(inc[i].cpu().data.numpy()[0])
        plot_arr.append(inc[i])

  print('Accuracy of the model on the 10000 test images: %d %%' % (
      100 * correct / total))
  
  return plot_arr, pre
