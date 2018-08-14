#pylint: disable=E1101
import os
import os.path
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
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resNet import Model
from functions import filter_size, train, test
from rescale import RandomResizedCrop, RandomRescale

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=100
learning_rate = 0.00001
batch_size = 128
batch_log = 70


train_transf = transforms.Compose([RandomRescale(size = 28, scales = (0.3, 1), sampling="uniform"), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
valid_transf = transforms.Compose([RandomRescale(size = 28, scales = (0.3, 1), sampling="uniform"), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
test_transf= transforms.Compose([RandomRescale(size = 28, scales = (0.3, 1), sampling="uniform"), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

root = './mnistdata'
if not os.path.exists(root):
    os.mkdir(root)

train_set = datasets.MNIST(root=root, train=True, transform=train_transf, download=True)
valid_set = datasets.MNIST(root=root, train=True, transform=valid_transf, download=True)

idx = list(range(len(train_set)))
np.random.seed(11)
np.random.shuffle(idx)
train_idx, valid_idx = idx[20000:], idx[:20000] #validation set of size 20'000
train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=1, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler, shuffle=False, num_workers=1, pin_memory=True)

test_set = datasets.MNIST(root=root, train=False, transform=test_transf, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

"""
transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalization for RGB
train_set = datasets.CIFAR10(root='./cifardata', train = True, transform = transform, download = True)

#test_set = datasets.CIFAR10(root='./cifardata', train=False, transform = transform, download=True)

resize = transforms.Compose([transforms.Pad(32), RandomResizedCrop(64, (0.05, 1)),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=resize,download=True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
#test_loader = DataLoader(test_set, batch_size = batch_size,shuffle = False)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""

f_in = 1

size, ratio, nratio = 5, 2**(1/3), 6
resnet1 = Model(f_in, size, ratio, nratio, srange=2)
resnet1.to(device)

f_size = filter_size(size, ratio, nratio)
resnet2 = Model(f_in, f_size, ratio**(-1), nratio, srange=2)
resnet2.to(device)

size, ratio, nratio = 5, 2**(2/3), 3
resnet3 = Model(f_in, size, ratio, nratio, srange=2)
resnet3.to(device)

f_size = filter_size(size, ratio, nratio)
resnet4 = Model(f_in, f_size, ratio**(-1), nratio, srange=2)
resnet4.to(device)

criterion = nn.CrossEntropyLoss()

train_loss1 =[]
train_acc1 = []
valid_loss1 = []
valid_acc1 = []

train_loss2 =[]
train_acc2 = []
valid_loss2 = []
valid_acc2 = []

train_loss3 =[]
train_acc3 = []
valid_loss3 = []
valid_acc3 = []


train_loss4 =[]
train_acc4 = []
valid_loss4 = []
valid_acc4 = []


for epoch in range(1, nb_epochs + 1):  
      
    train_l, train_a = train(resnet2, train_loader, learning_rate, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet2, train_loader, criterion, epoch, batch_log, device) 
    train_loss2.append(train_l)
    train_acc2.append(train_a)

    valid_l, valid_a = test(resnet2, valid_loader, criterion, epoch, batch_log, device) 
    valid_loss2.append(valid_l)
    valid_acc2.append(valid_a)

with open("compareResnet_log2.txt", "w") as output:
    output.write("ResNet\t size=16\t ratio=2^(-1/3)\t nratio=6 \n")
    output.write(str(train_loss2))
    output.write(str(train_acc2))
    output.write(str(valid_loss2))
    output.write(str(valid_acc2))

for epoch in range(1, nb_epochs + 1):  
      
    train_l, train_a = train(resnet1, train_loader, learning_rate, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet1, train_loader, criterion, epoch, batch_log, device) 
    train_loss1.append(train_l)
    train_acc1.append(train_a)

    valid_l, valid_a = test(resnet1, valid_loader, criterion, epoch, batch_log, device) 
    valid_loss1.append(valid_l)
    valid_acc1.append(valid_a)

with open("compareResnet_log1.txt", "w") as output:
    output.write("nb_epochs=100\t learning_rate = 0.00001\t batch_size = 128 \n")
    output.write("ResNet\t size=5\t ratio=2^(1/3)\t nratio=6 \n")
    output.write(str(train_loss1))
    output.write(str(train_acc1))
    output.write(str(valid_loss1))
    output.write(str(valid_acc1))




for epoch in range(1, nb_epochs + 1):  
      
    train_l, train_a = train(resnet3, train_loader, learning_rate, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet3, train_loader, criterion, epoch, batch_log, device) 
    train_loss3.append(train_l)
    train_acc3.append(train_a)

    valid_l, valid_a = test(resnet3, valid_loader, criterion, epoch, batch_log, device) 
    valid_loss3.append(valid_l)
    valid_acc3.append(valid_a)

with open("compareResnet_log3.txt", "w") as output:
    output.write("ResNet\t size=5\t ratio=2^(2/3)\t nratio=3 \n")
    output.write(str(train_loss3))
    output.write(str(train_acc3))
    output.write(str(valid_loss3))
    output.write(str(valid_acc3))


for epoch in range(1, nb_epochs + 1):  
      
    train_l, train_a = train(resnet4, train_loader, learning_rate, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet4, train_loader, criterion, epoch, batch_log, device) 
    train_loss4.append(train_l)
    train_acc4.append(train_a)

    valid_l, valid_a = test(resnet4, valid_loader, criterion, epoch, batch_log, device) 
    valid_loss4.append(valid_l)
    valid_acc4.append(valid_a)

with open("compareResnet_log4.txt", "w") as output:
    output.write("ResNet\t size=13\t ratio=2^(-2/3)\t nratio=3 \n")
    output.write(str(train_loss4))
    output.write(str(train_acc4))
    output.write(str(valid_loss4))
    output.write(str(valid_acc4))



plt.figure()
plt.plot(train_loss1, label = "ResNet size 5, rat 2^(1/3), 6")
plt.plot(train_loss2, label = "ResNet size 16, rat 2^(-1/3), 6")
plt.plot(train_loss3, label = "ResNet size 5, rat 2^(2/3), 3")
plt.plot(train_loss4, label = "ResNet size 13, rat 2^(-2/3), 3")
plt.title("Training loss Scaled MNIST")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_resnet_comp_mnist.pdf")
#plt.show()

plt.figure()
plt.plot(train_acc1, label = "ResNet size 5, rat 2^(1/3), 6")
plt.plot(train_acc2, label = "ResNet size 16, rat 2^(-1/3), 6")
plt.plot(train_acc3, label = "ResNet size 5, rat 2^(2/3), 3")
plt.plot(train_acc4, label = "ResNet size 13, rat 2^(-2/3), 3")
plt.title("Training accuracy Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_resnet_comp_mnist.pdf")
#plt.show()

plt.figure() 
plt.plot(valid_loss1, label = "ResNet size 9, rat 2^(-1/3), 6")
plt.plot(valid_loss2, label = "ResNet size 16, rat 2^(-1/3), 6")
plt.plot(valid_loss3, label = "ResNet size 3, rat 2^(1/3), 6")
plt.plot(valid_loss4, label = "ResNet size 7, rat 2^(1/3), 6")
plt.title("Validation loss Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("valid_loss_resnet_comp_mnist.pdf")
#plt.show()

plt.figure()
plt.plot(valid_acc1, label = "ResNet size 9, rat 2^(-1/3), 6")
plt.plot(valid_acc2, label = "ResNet size 16, rat 2^(-1/3), 6")
plt.plot(valid_acc3, label = "ResNet size 3, rat 2^(1/3), 6")
plt.plot(valid_acc4, label = "ResNet size 7, rat 2^(1/3), 6")
plt.title("Validation accuracy Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("valid_acc_resnet_comp_mnist.pdf")
#plt.show()