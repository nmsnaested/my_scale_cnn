#pylint: disable=E1101

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from baseCNNs import SiCNN

from functions import train, test
from rescale import RandomResizedCrop

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=100
learning_rate = 0.00001
batch_size = 128
batch_log = 50

transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.CIFAR10(root='./cifardata', train=True, transform=transform, download=True)

#test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=transform,download=True)

resize = transforms.Compose([transforms.Pad(32), RandomResizedCrop(size = 64, scale = (0.05, 1)), 
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=resize,download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


net1 = SiCNN(2**(1/3), 8)
net1.to(device)

net2 = SiCNN(2**(2/3), 3)
net2.to(device)

net3 = SiCNN(2**(2/3), 6)
net3.to(device)

net4 = SiCNN(2**0.5, 6)
net4.to(device)

net5 = SiCNN(2**0.5, 8)
net5.to(device)



criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate)
optimizer3 = optim.Adam(net3.parameters(), lr=learning_rate)
optimizer4 = optim.Adam(net4.parameters(), lr=learning_rate)
optimizer5 = optim.Adam(net5.parameters(), lr=learning_rate)

train_loss1 = []
train_acc1 = []
test_loss1 = []
test_acc1 = []

for epoch in range(1, nb_epochs + 1):  
    train_l,train_a = train(net1, train_loader, optimizer1, criterion, epoch, batch_log, device) 
    train_l,train_a = test(net1, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(net1, test_loader, criterion, epoch, batch_log, device)
    train_loss1.append(train_l)
    train_acc1.append(train_a)
    test_loss1.append(test_l)
    test_acc1.append(test_a)

with open("compareRatios_log1.txt", "w") as output:
    output.write("nb_epochs=100\t learning_rate = 0.00001\t batch_size = 128 \n")
    output.write("SiCNN1\t ratio=2^(1/3)\t nratio=8 \n")
    output.write(str(train_loss1))
    output.write(str(train_acc1))
    output.write(str(test_loss1))
    output.write(str(test_acc1))

train_loss2 = []
train_acc2 = []
test_loss2 = []
test_acc2 = []

for epoch in range(1, nb_epochs + 1):  
    train_l,train_a = train(net2, train_loader, optimizer2, criterion, epoch, batch_log, device) 
    train_l,train_a = test(net2, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(net2, test_loader, criterion, epoch, batch_log, device)
    train_loss2.append(train_l)
    train_acc2.append(train_a)
    test_loss2.append(test_l)
    test_acc2.append(test_a)

with open("compareRatios_log2.txt", "w") as output:
    output.write("SiCNN2\t ratio=2^(2/3)\t nratio=3 \n")
    output.write(str(train_loss2))
    output.write(str(train_acc2))
    output.write(str(test_loss2))
    output.write(str(test_acc2))


train_loss3 = []
train_acc3 = []
test_loss3 = []
test_acc3 = []

for epoch in range(1, nb_epochs + 1):  
    train_l,train_a = train(net3, train_loader, optimizer3, criterion, epoch, batch_log, device) 
    train_l,train_a = test(net3, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(net3, test_loader, criterion, epoch, batch_log, device)
    train_loss3.append(train_l)
    train_acc3.append(train_a)
    test_loss3.append(test_l)
    test_acc3.append(test_a)

with open("compareRatios_log3.txt", "w") as output:
    output.write("SiCNN3\t ratio=2^(2/3)\t nratio=6 \n")
    output.write(str(train_loss3))
    output.write(str(train_acc3))
    output.write(str(test_loss3))
    output.write(str(test_acc3))


train_loss4 = []
train_acc4 = []
test_loss4 = []
test_acc4 = []

for epoch in range(1, nb_epochs + 1):  
    train_l,train_a = train(net4, train_loader, optimizer4, criterion, epoch, batch_log, device) 
    train_l,train_a = test(net4, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(net4, test_loader, criterion, epoch, batch_log, device)
    train_loss4.append(train_l)
    train_acc4.append(train_a)
    test_loss4.append(test_l)
    test_acc4.append(test_a)

with open("compareRatios_log4.txt", "w") as output:
    output.write("SiCNN4\t ratio=2^(1/2)\t nratio=6 \n")
    output.write(str(train_loss4))
    output.write(str(train_acc4))
    output.write(str(test_loss4))
    output.write(str(test_acc4))


train_loss5 = []
train_acc5 = []
test_loss5 = []
test_acc5 = []

for epoch in range(1, nb_epochs + 1):  
    train_l,train_a = train(net5, train_loader, optimizer5, criterion, epoch, batch_log, device) 
    train_l,train_a = test(net5, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(net5, test_loader, criterion, epoch, batch_log, device)
    train_loss5.append(train_l)
    train_acc5.append(train_a)
    test_loss5.append(test_l)
    test_acc5.append(test_a)

with open("compareRatios_log5.txt", "w") as output:
    output.write("SiCNN5\t ratio=2^(1/2)\t nratio=8 \n")
    output.write(str(train_loss5))
    output.write(str(train_acc5))
    output.write(str(test_loss5))
    output.write(str(test_acc5))


plt.figure()
plt.plot(train_loss1, label = "SiCNN 2^(1/3), 8")
plt.plot(train_loss2, label = "SiCNN 2^(2/3), 3")
plt.plot(train_loss3, label = "SiCNN 2^(2/3), 6")
plt.plot(train_loss4, label = "SiCNN 2^(1/2), 6")
plt.plot(train_loss5, label = "SiCNN 2^(1/2), 8")
plt.title("Training loss CIFAR10")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_compare_ratios.pdf")
#plt.show()

plt.figure()
plt.plot(train_acc1, label = "SiCNN 2^(1/3), 8")
plt.plot(train_acc2, label = "SiCNN 2^(2/3), 3")
plt.plot(train_acc3, label = "SiCNN 2^(2/3), 6")
plt.plot(train_acc4, label = "SiCNN 2^(1/2), 6")
plt.plot(train_acc5, label = "SiCNN 2^(1/2), 8")
plt.title("Training accuracy CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_compare_ratios.pdf")
#plt.show()

plt.figure()
plt.plot(test_loss1, label = "SiCNN 2^(1/3), 8")
plt.plot(test_loss2, label = "SiCNN 2^(2/3), 3")
plt.plot(test_loss3, label = "SiCNN 2^(2/3), 6")
plt.plot(test_loss4, label = "SiCNN 2^(1/2), 6")
plt.plot(test_loss5, label = "SiCNN 2^(1/2), 8")
plt.title("Test loss Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_compare_ratios.pdf")
#plt.show()

plt.figure()
plt.plot(test_acc1, label = "SiCNN 2^(1/3), 8")
plt.plot(test_acc2, label = "SiCNN 2^(2/3), 3")
plt.plot(test_acc3, label = "SiCNN 2^(2/3), 6")
plt.plot(test_acc4, label = "SiCNN 2^(1/2), 6")
plt.plot(test_acc5, label = "SiCNN 2^(1/2), 8")
plt.title("Test accuracy Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_compare_ratios.pdf")
#plt.show()
