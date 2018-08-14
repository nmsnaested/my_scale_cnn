#pylint: disable=E1101

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resNet import Model
from functions import train, test
from rescale import RandomResizedCrop

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=200
learning_rate = 0.00001
batch_size = 128
batch_log = 200

RATIO = 2**(-1/3)
NRATIO = 6
SIZE = 9
PADDING = 1
SRANGE = 2

parameters = {
    "nb_epochs": nb_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "ratio": RATIO,
    "nb_channels": NRATIO    
}
phandle = open("ResNet_long_log.pickle", "wb")
pickle.dump(parameters, phandle)

transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalization for RGB
train_set = datasets.CIFAR10(root='./cifardata', train = True, transform = transform, download = True)

#test_set = datasets.CIFAR10(root='./cifardata', train=False, transform = transform, download=True)

resize = transforms.Compose([transforms.Pad(32), RandomResizedCrop(64, (0.05, 1)),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set_sc = datasets.CIFAR10(root='./cifardata',train=False,transform=resize,download=True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
#test_loader = DataLoader(test_set, batch_size = batch_size,shuffle = False)
test_loader_sc = DataLoader(test_set_sc, batch_size = batch_size, shuffle = False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

resnet = Model(SIZE, RATIO, NRATIO, SRANGE, PADDING)
resnet.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet.parameters(), lr = learning_rate)

all_train_loss=[]
all_train_acc = []
all_test_loss = []
all_test_acc = []

dynamics = []

for epoch in range(1, nb_epochs + 1):  
      
    train_l, train_a = train(resnet, train_loader, optimizer, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet, train_loader, criterion, epoch, batch_log, device) 
    all_train_loss.append(train_l)
    all_train_acc.append(train_a)

    test_l, test_a = test(resnet, test_loader_sc, criterion, epoch, batch_log, device) 
    all_test_loss.append(test_l)
    all_test_acc.append(test_a)

    dynamics.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

pickle.dump(dynamics, phandle)
phandle.close()

plt.figure()
plt.plot(all_train_loss, label = "ResNet")
plt.title("Training loss CIFAR10")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_resnet_lr10-5_long.pdf")
#plt.show()

plt.figure()
plt.plot(all_train_acc, label = "ResNet")
plt.title("Training accuracy CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_resnet_lr10-5_long.pdf")
#plt.show()

plt.figure() 
plt.plot(all_test_loss, label = "ResNet")
plt.title("Test loss Scaled CIFAR 10")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_resnet_scaled_lr10-5_long.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_acc, label = "ResNet CIFAR10")
plt.title("Test accuracy Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_resnet_scaled_lr10-5_long.pdf")
#plt.show()