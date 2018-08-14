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

nb_epochs=150
learning_rate = 0.0001
batch_size = 128
batch_log = 70
repeats = 6

f_in = 1
size=5
ratio=2**(2/3)
nratio=3
srange=2
padding=0

log = open("mnist_mean_log.pickle", "wb")

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "repetitions": repeats,
    "size": size, 
    "ratio": ratio,
    "nratio": nratio,
    "srange": srange
}
pickle.dump(parameters, log)

uniform = transforms.Compose([RandomRescale(size = 28, scales = (0.3, 1), sampling = "uniform"), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

root = './mnistdata'
if not os.path.exists(root):
    os.mkdir(root)

idx = list(range(60000))
train_set = Subset(datasets.MNIST(root=root, train=True, transform=uniform, download=True), idx[20000:])
valid_set = Subset(datasets.MNIST(root=root, train=True, transform=uniform, download=True), idx[:20000])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

test_set = datasets.MNIST(root=root, train=False, transform=uniform, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()

pickle.dump(repeats, log)

for ii in range(repeats): 
    model = SiCNN_3(f_in, size, ratio, nratio, srange, padding)
    model.to(device)

    train_loss=[]
    train_acc = []
    valid_loss = []
    valid_acc = []

    for epoch in range(1, nb_epochs + 1): 
        train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
        train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
        valid_l, valid_a = test(model, valid_loader, criterion, epoch, batch_log, device)
        train_loss.append(train_l)
        train_acc.append(train_a) 
        valid_loss.append(valid_l)
        valid_acc.append(valid_a)

    dynamics = {
        "model": ii,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc
    }
    pickle.dump(dynamics, log)

log.close()

plot_figures("mnist_mean_log.pickle", name="MNIST", mean = True)