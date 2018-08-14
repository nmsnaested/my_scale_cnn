#pylint: disable=E1101
import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torchvision

import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler

import loaddataset as lds

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from mnistCNNs import SiCNN, SiCNN2, SiCNN3, miniSiAll

from functions import train, test
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=200
learning_rate = 0.0001
batch_size = 256
batch_log = 70

train_transf = transforms.Compose([RandomRescale(size = 28, scale = (0.3, 1)), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
valid_transf = transforms.Compose([RandomRescale(size = 28, scale = (0.3, 1)), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
test_transf= transforms.Compose([RandomRescale(size = 28, scale = (0.3, 1)), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

root = './mnistdata'
if not os.path.exists(root):
    os.mkdir(root)

train_set = datasets.MNIST(root=root, train=True, transform=train_transf, download=True)
valid_set = datasets.MNIST(root=root, train=True, transform=valid_transf, download=True)

idx = list(range(len(train_set)))
np.random.seed(11)
np.random.shuffle(idx)
train_idx = idx[20000:]
valid_idx = idx[:20000] #validation set of size 20'000
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=1, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, sampler=valid_sampler, shuffle=False, num_workers=1, pin_memory=True)

test_set = datasets.MNIST(root=root, train=False, transform=test_transf, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False)

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

ratio = 2**(2/3)
nratio = 3

sicnn1 = SiCNN(ratio, nratio)
sicnn1.to(device)

sicnn2 = SiCNN2(ratio, nratio)
sicnn2.to(device)

sicnn3 = SiCNN3(ratio, nratio)
sicnn3.to(device)

mini = miniSiAll(ratio, nratio)
mini.to(device)


criterion = nn.CrossEntropyLoss()

optimizer1 = optim.Adam(sicnn1.parameters(), lr = learning_rate)

optimizer2 = optim.Adam(sicnn2.parameters(), lr = learning_rate)
#scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=40, gamma=0.1)

optimizer3 = optim.Adam(sicnn3.parameters(), lr = learning_rate)


optimizerm = optim.Adam(mini.parameters(), lr = learning_rate)

train_loss_1=[]
train_acc_1 = []
valid_loss_1 = []
valid_acc_1 = []

for epoch in range(1, nb_epochs + 1): 
    
    train_l, train_a = train(sicnn1, train_loader, optimizer1, criterion, epoch, batch_log, device) 
    train_l, train_a = test(sicnn1, train_loader, criterion, epoch, batch_log, device) 
    valid_l, valid_a = test(sicnn1, valid_loader, criterion, epoch, batch_log, device)
    train_loss_1.append(train_l)
    train_acc_1.append(train_a) 
    valid_loss_1.append(valid_l)
    valid_acc_1.append(valid_a)

with open("mnist_sicnn1_log.txt", "w") as output:
    output.write("nb_epochs=200\t lr=0.0001 \t batch_size=256\n")
    output.write("SiCNN kanazawa \t ratio=2^(2/3), nratio=3, srange=3\n")
    output.write(str(train_loss_1))
    output.write("\n")
    output.write(str(train_acc_1))
    output.write("\n")
    output.write(str(valid_loss_1))
    output.write("\n")
    output.write(str(valid_acc_1))

train_loss_2=[]
train_acc_2 = []
valid_loss_2 = []
valid_acc_2 = []

for epoch in range(1, nb_epochs + 1): 
    #scheduler.step() 
    train_l, train_a = train(sicnn2, train_loader, optimizer2, criterion, epoch, batch_log, device) 
    train_l, train_a = test(sicnn2, train_loader, criterion, epoch, batch_log, device) 
    valid_l, valid_a = test(sicnn2, valid_loader, criterion, epoch, batch_log, device)
    train_loss_2.append(train_l)
    train_acc_2.append(train_a)
    valid_loss_2.append(valid_l)
    valid_acc_2.append(valid_a)

with open("mnist_sicnn2_log.txt", "w") as output:
    output.write("SiCNN 3 ConvLayers 'other' \t ratio=2^(2/3), nratio=3 \n")
    output.write(str(train_loss_2))
    output.write("\n")
    output.write(str(train_acc_2))
    output.write("\n")
    output.write(str(valid_loss_2))
    output.write("\n")
    output.write(str(valid_acc_2))


train_loss_3=[]
train_acc_3 = []
valid_loss_3 = []
valid_acc_3 = []

for epoch in range(1, nb_epochs + 1): 
    #scheduler.step() 
    train_l, train_a = train(sicnn3, train_loader, optimizer3, criterion, epoch, batch_log, device) 
    train_l, train_a = test(sicnn3, train_loader, criterion, epoch, batch_log, device) 
    valid_l, valid_a = test(sicnn3, valid_loader, criterion, epoch, batch_log, device)
    train_loss_3.append(train_l)
    train_acc_3.append(train_a)
    valid_loss_3.append(valid_l)
    valid_acc_3.append(valid_a)

with open("mnist_sicnn3_log.txt", "w") as output:
    output.write("SiCNN 3 ConvLayers 'other' \t ratio=2^(2/3), nratio=3 \t srange=2\n")
    output.write(str(train_loss_3))
    output.write("\n")
    output.write(str(train_acc_3))
    output.write("\n")
    output.write(str(valid_loss_3))
    output.write("\n")
    output.write(str(valid_acc_3))

train_loss_m=[]
train_acc_m = []
valid_loss_m = []
valid_acc_m = []

for epoch in range(1, nb_epochs + 1): 
    train_l, train_a = train(mini, train_loader, optimizerm, criterion, epoch, batch_log, device) 
    train_l, train_a = test(mini, train_loader, criterion, epoch, batch_log, device) 
    valid_l, valid_a = test(mini, valid_loader, criterion, epoch, batch_log, device)
    train_loss_m.append(train_l)
    train_acc_m.append(train_a)
    valid_loss_m.append(valid_l)
    valid_acc_m.append(valid_a)

with open("mnist_mini_log.txt", "w") as output:
    output.write("small AllCNN \t ratio=2^(2/3), nratio=3, srange=3 \n")
    output.write(str(train_loss_m))
    output.write("\n")
    output.write(str(train_acc_m))
    output.write("\n")
    output.write(str(valid_loss_m))
    output.write("\n")
    output.write(str(valid_acc_m))

plt.figure()
plt.plot(train_loss_1, label = "SiCNN K. sr=3")
plt.plot(train_loss_2, label = "SiCNN 3 CL")
plt.plot(train_loss_3, label = "SiCNN 3 CL sr=2")
plt.plot(train_loss_m, label = "small Si-AllCNN")
plt.title("Training loss MNIST")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_mnist_comp_1.pdf")
#plt.show()

plt.figure()
plt.plot(train_acc_1, label = "SiCNN K. sr=3")
plt.plot(train_acc_2, label = "SiCNN 3 CL")
plt.plot(train_acc_3, label = "SiCNN 3 CL sr=2")
plt.plot(train_acc_m, label = "small Si-AllCNN")
plt.title("Training accuracy MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_mnist_comp_1.pdf")
#plt.show()

plt.figure()
plt.plot(valid_loss_1, label = "SiCNN K. sr=3")
plt.plot(valid_loss_2, label = "SiCNN 3 CL")
plt.plot(valid_loss_3, label = "SiCNN 3 CL sr=2")
plt.plot(valid_loss_m, label = "small Si-AllCNN")
plt.title("Validation loss Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("valid_loss_mnist_scaled_comp_1.pdf")
#plt.show()

plt.figure()
plt.plot(valid_acc_1, label = "SiCNN K. sr=3")
plt.plot(valid_acc_2, label = "SiCNN 3 CL")
plt.plot(valid_acc_3, label = "SiCNN 3 CL sr=2")
plt.plot(valid_acc_m, label = "small Si-AllCNN")
plt.title("Validation accuracy Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("valid_acc_mnist_scaled_comp_1.pdf")
