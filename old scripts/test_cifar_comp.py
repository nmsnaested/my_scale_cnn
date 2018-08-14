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

from compareCNNs import SiCNN1, SiCNN2, SiCNN3, miniSiAll

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

ratio = 2**(2/3)
nratio = 3

net1 = SiCNN1(ratio, nratio)
net1.to(device)

net2 = SiCNN2(ratio, nratio)
net2.to(device)

net3 = SiCNN3(ratio, nratio)
net3.to(device)

net4 = miniSiAll(ratio, nratio)
net4.to(device)


criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate)
optimizer3 = optim.Adam(net3.parameters(), lr=learning_rate)
optimizer4 = optim.Adam(net4.parameters(), lr=learning_rate)



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

with open("compareCNNs_log1.txt", "w") as output:
    output.write("nb_epochs=100\t learning_rate=0.00001\t batch_size=128\t ratio=2^(2/3)\t nratio=3 \n")
    output.write("SiCNN1\t 3,36,3-36,64,3-64,150,10\tsrange=0,3\n")
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

with open("compareCNNs_log2.txt", "w") as output:
    output.write("SiCNN2\t 3,48,5-48,92,5-92,92,5-92,150,10\tsrange=0,1,1\n")
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

with open("compareCNNs_log3.txt", "w") as output:
    output.write("SiCNN3\t 3,96,5-96,96,3-96,192,3-192,150,10\tsrange=0,1,1\n")
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

with open("compareCNNs_log4.txt", "w") as output:
    output.write("mini SiAllCNN\t 3,96,3-96,96,3-96,192,3-192,192,3-192,10\tsrange=0,2,2,2\n")
    output.write(str(train_loss4))
    output.write(str(train_acc4))
    output.write(str(test_loss4))
    output.write(str(test_acc4))

plt.figure()
plt.plot(train_loss1, label = "K. srange = 3")
plt.plot(train_loss2, label = "3 ConvLayers")
plt.plot(train_loss3, label = "3 ConvL other")
plt.plot(train_loss4, label = "small Si-AllCNN")
plt.title("Training loss CIFAR10")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_compare_sicnns_4.pdf")
#plt.show()

plt.figure()
plt.plot(train_acc1, label = "K. srange = 3")
plt.plot(train_acc2, label = "3 ConvLayers")
plt.plot(train_acc3, label = "3 ConvL other")
plt.plot(train_acc4, label = "small Si-AllCNN")
plt.title("Training accuracy CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_compare_sicnns_4.pdf")
#plt.show()

plt.figure()
plt.plot(test_loss1, label = "K. srange = 3")
plt.plot(test_loss2, label = "3 ConvLayers")
plt.plot(test_loss3, label = "3 ConvL other")
plt.plot(test_loss4, label = "small Si-AllCNN")
plt.title("Test loss Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_compare_sicnns_4.pdf")
#plt.show()

plt.figure()
plt.plot(test_acc1, label = "K. srange = 3")
plt.plot(test_acc2, label = "3 ConvLayers")
plt.plot(test_acc3, label = "3 ConvL other")
plt.plot(test_acc4, label = "small Si-AllCNN")
plt.title("Test accuracy Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_compare_sicnns_4.pdf")
#plt.show()
