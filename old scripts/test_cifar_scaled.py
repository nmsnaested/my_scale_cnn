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

from architectures import StdNet, SiCNN
from functions import train, test
from rescale import RandomResizedCrop
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=70
learning_rate = 0.00001
batch_size = 256
batch_log = 50

ratio = (2**(1/3))
nratio = 6

parameters = {
    "nb_epochs": nb_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "ratio": ratio,
    "nb_channels": nratio    
}
phandle = open("ScaledCifar10_log.pickle", "wb")
pickle.dump(parameters, phandle)

transform = transforms.Compose([transforms.Resize(64), 
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #mean 0 std 1 for RGB
train_set = datasets.CIFAR10(root = './cifardata', train = True, transform = transform, download = True)

#test_set = datasets.CIFAR10(root = './cifardata', train = False, transform = transform, download = True)

resize = transforms.Compose([transforms.Pad(32), RandomResizedCrop(size = 64, scale = (0.05, 1)), 
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=resize,download=True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
#Show images of train or test set 

dataiter = iter(test_loader) 
images, labels = dataiter.next()

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
img = torchvision.utils.make_grid(images)
img = img / 2 + 0.5     # unnormalize
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

"""
stdnet = StdNet(f_in=3)
stdnet.to(device)

sicnn = SiCNN(f_in=3, ratio=ratio, nratio=nratio)
sicnn.to(device)

criterion = nn.CrossEntropyLoss()
std_optimizer = optim.Adam(stdnet.parameters(), lr = learning_rate)
sicnn_optimizer = optim.Adam(sicnn.parameters(), lr = learning_rate)

sicnn_train_loss=[]
sicnn_train_acc = []
sicnn_test_loss = []
sicnn_test_acc = []

sicnn_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(sicnn, train_loader, sicnn_optimizer, criterion, epoch, batch_log, device) 
    test_l, test_a = test(sicnn, test_loader, criterion, epoch, batch_log, device)
    sicnn_train_loss.append(train_l)
    sicnn_train_acc.append(train_a)
    sicnn_test_loss.append(test_l)
    sicnn_test_acc.append(test_a)

    sicnn_dyn.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

std_train_loss = []
std_train_acc = []
std_test_loss = []
std_test_acc = []

std_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(stdnet, train_loader, std_optimizer, criterion, epoch, batch_log, device) 
    test_l, test_a = test(stdnet, test_loader, criterion, epoch, batch_log, device)
    std_train_loss.append(train_l)
    std_train_acc.append(train_a)
    std_test_loss.append(test_l)
    std_test_acc.append(test_a)

    std_dyn.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

pickle.dump({"sicnn_dyn": sicnn_dyn, "stdcnn_dyn": std_dyn}, phandle)
phandle.close()

plt.figure()
plt.plot(std_train_loss, label = "Standard ConvNet")
plt.plot(sicnn_train_loss, label = "SiCNN")
plt.title("Training loss Scaled CIFAR10, batch size 256, lr=0.00001")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_cifar10_scaled_256_00001.pdf")
#plt.show()

plt.figure()
plt.plot(std_train_acc, label = "Standard ConvNet")
plt.plot(sicnn_train_acc, label = "SiCNN")
plt.title("Training accuracy Scaled CIFAR10, batch size 256, lr=0.00001")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_cifar10_scaled_256_00001.pdf")
#plt.show()

plt.figure()
plt.plot(std_test_loss, label = "Standard ConvNet")
plt.plot(sicnn_test_loss, label = "SiCNN")
plt.title("Test loss Scaled CIFAR10, batch size 256, lr=0.00001")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_cifar10_scaled_256_00001.pdf")
#plt.show()

plt.figure()
plt.plot(std_test_acc, label = "Standard ConvNet")
plt.plot(sicnn_test_acc, label = "SiCNN")
plt.title("Test accuracy Scaled CIFAR10, batch size 256, lr=0.00001")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_cifar10_scaled_256_00001.pdf")
#plt.show()
