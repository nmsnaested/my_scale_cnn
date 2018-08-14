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

from allCNNs import AllCNN, SiAllCNN
from functions import train, test
from rescale import RandomResizedCrop

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs = 50
learning_rate = 0.0001
batch_size = 32
batch_log = 100

ratio = 2**(1/3)
nratio = 6

transform = transforms.Compose([transforms.Resize(64), 
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalization for RGB
train_set = datasets.CIFAR10(root='./cifardata', train=True, transform = transform, download = True)

test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=transform,download=True)

resize = transforms.Compose([transforms.Pad(32), RandomResizedCrop(64, (0.05, 1)), 
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set_sc = datasets.CIFAR10(root='./cifardata', train = False, transform = resize, download = True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
test_loader_sc = DataLoader(test_set_sc, batch_size = batch_size, shuffle = False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

allcnn = AllCNN()
allcnn.to(device)

siallcnn = SiAllCNN(ratio, nratio)
siallcnn.to(device)

criterion = nn.CrossEntropyLoss()

all_optimizer = optim.Adam(allcnn.parameters(), lr = learning_rate)
siall_optimizer = optim.Adam(siallcnn.parameters(), lr = learning_rate)

siall_train_loss = []
siall_train_acc = []
siall_test_loss = []
siall_test_acc = []
siall_test_loss_sc = []
siall_test_acc_sc = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(siallcnn, train_loader, siall_optimizer, criterion, epoch, batch_log, device) 
    test_l, test_a = test(siallcnn, test_loader, criterion, epoch, batch_log, device)
    siall_train_loss.append(train_l)
    siall_train_acc.append(train_a)
    siall_test_loss.append(test_l)
    siall_test_acc.append(test_a)
    test_l_sc, test_a_sc = test(siallcnn, test_loader_sc, criterion, epoch, batch_log, device)
    siall_test_loss_sc.append(test_l_sc)
    siall_test_acc_sc.append(test_a_sc)

with open("si_allcnn_log_scaled.txt", "w") as output:
    output.write(str(siall_train_loss))
    output.write(str(siall_train_acc))
    output.write(str(siall_test_loss))
    output.write(str(siall_test_acc))
    output.write(str(siall_test_loss_sc))
    output.write(str(siall_test_acc_sc))

all_train_loss = []
all_train_acc = []
all_test_loss = []
all_test_acc = []
all_test_loss_sc = []
all_test_acc_sc = []

for epoch in range(1, nb_epochs+1):  
    train_l, train_a = train(allcnn, train_loader, all_optimizer, criterion, epoch, batch_log, device) 
    test_l, test_a = test(allcnn, test_loader, criterion, epoch, batch_log, device)
    all_train_loss.append(train_l)
    all_train_acc.append(train_a)
    all_test_loss.append(test_l)
    all_test_acc.append(test_a)
    test_l_sc, test_a_sc = test(allcnn, test_loader_sc, criterion, epoch, batch_log, device)
    all_test_loss_sc.append(test_l_sc)
    all_test_acc_sc.append(test_a_sc)

with open("std_allcnn_log_scaled.txt", "w") as output:
    output.write(str(all_train_loss))
    output.write(str(all_train_acc))
    output.write(str(all_test_loss))
    output.write(str(all_test_acc))
    output.write(str(all_test_loss_sc))
    output.write(str(all_test_acc_sc))

plt.figure()
plt.plot(all_train_loss, label = "AllCNN")
plt.plot(siall_train_loss, label = "Si-AllCNN")
plt.title("Training loss CIFAR10")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_allcnn.pdf")
#plt.show()

plt.figure()
plt.plot(all_train_acc, label = "AllCNN")
plt.plot(siall_train_acc, label = "Si-AllCNN")
plt.title("Training accuracy CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_allcnn.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_loss, label = "AllCNN")
plt.plot(siall_test_loss, label = "Si-AllCNN")
plt.title("Test loss CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_allcnn.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_acc, label = "AllCNN")
plt.plot(siall_test_acc, label = "Si-AllCNN")
plt.title("Test accuracy CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_allcnn.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_loss_sc, label = "AllCNN")
plt.plot(siall_test_loss_sc, label = "Si-AllCNN")
plt.title("Test loss Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_allcnn_scaled.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_acc_sc, label = "AllCNN")
plt.plot(siall_test_acc_sc, label = "Si-AllCNN")
plt.title("Test accuracy Scaled CIFAR10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_allcnn_scaled.pdf")
#plt.show()