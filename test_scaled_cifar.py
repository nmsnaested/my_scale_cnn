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

from architectures import SiCNN_3
from functions import train, test, plot_figures
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

nb_epochs=100
learning_rate = 0.00001
batch_size = 128
batch_log = 70
repeats = 6

f_in = 3
size = 5
ratio = 2**(2/3)
nratio = 3
srange = 2

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "filter size": size,
    "ratio": ratio,
    "nb channels": nratio,
    "overlap": srange    
}
log = open("cifar10_mean_log.pickle", "wb")
pickle.dump(parameters, log)

scales = [0.75, 0.875, 1.0] # crop center 24x24 or 28x28 and resize to 32x32, or leave 32x32 

train_transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #mean 0 std 1 for RGB


idx = list(range(50000))
train_set = Subset(datasets.CIFAR10(root='./cifardata', train=True, transform=train_transf, download=True), idx[10000:])

#test_set = datasets.CIFAR10(root='./cifardata', train=False, transform=test_transf, download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()

pickle.dump(repeats, log)

for ii in range(repeats): 
    model = SiCNN_3(f_in, size, ratio, nratio, srange)
    model.to(device)

    train_loss=[]
    train_acc = []
    valid_loss = []
    valid_acc = []

    for epoch in range(1, nb_epochs + 1): 
        train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
        train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
        train_loss.append(train_l)
        train_acc.append(train_a) 
    
    dynamics = {
        "model": ii,
        "train_loss": train_loss,
        "train_acc": train_acc
    }
    pickle.dump(dynamics, log)
    with open("trained_model_{}.pickle".format(ii), "wb") as save:
        pickle.dump(model, save)

plot_figures("cifar10_mean_log.pickle", name="CIFAR10", train=True, mean=True)

log.close()

log = open("cifar10_valid_mean_log.pickle", "wb")
pickle.dump(parameters, log)
pickle.dump(repeats * len(scales), log)

for ii in range(repeats):
    model = pickle.load("trained_model_{}.pickle".format(ii), "rb")

    for s in scales: 
        test_transf = transforms.Compose([
            RandomRescale(size=32, sampling="uniform", scales=(s, s)), 
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) 
        valid_set = Subset(datasets.CIFAR10(root='./cifardata', train=True, transform=test_transf, download=True), idx[:10000])
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        s_valid_loss = []
        s_valid_acc = []
        for epoch in range(1, nb_epochs + 1):
            valid_l, valid_a = test(model, valid_loader, criterion, epoch, batch_log, device)
            s_valid_loss.append(valid_l)
            s_valid_acc.append(valid_a)

        dynamics = {
            "model": ii,
            "scale": s,
            "valid_loss": s_valid_loss,
            "valid_acc": s_valid_acc
        }
        pickle.dump(dynamics, log)    

plot_figures("cifar10_valid_mean_log.pickle", name="CIFAR10", train=False, mean=True)