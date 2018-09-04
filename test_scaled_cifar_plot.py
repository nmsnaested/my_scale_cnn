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

nb_epochs=200
learning_rate = 0.00001
batch_size = 128
batch_log = 70
repeats = 6

f_in = 3
size = 5
ratio = 2**(2/3)
nratio = 3
srange = 2

scales = [0.75, 0.875, 1.0] # crop center 24x24 or 28x28 and resize to 32x32, or leave 32x32 

criterion = nn.CrossEntropyLoss()

idx = list(range(50000))

avg_losses = []
std_losses = []
avg_accs = []
std_accs = []
for s in scales:
    test_transf = transforms.Compose([
    RandomRescale(size=32, sampling="uniform", scales=(s, s)), 
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    valid_set = Subset(datasets.CIFAR10(root='./cifardata', train=True, transform=test_transf, download=True), idx[:10000])
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    valid_loss = []
    valid_acc = []
    for ii in range(repeats):
        infile = open("model_{}_{}_cifar.pickle".format(s, ii), "rb")
        model = pickle.load(infile)
        epoch = 200
        valid_l, valid_a = test(model, valid_loader, criterion, epoch, batch_log, device)
        valid_loss.append(valid_l)
        valid_acc.append (valid_a)
    
    avg_losses.append(np.mean(np.array(valid_loss)))
    std_losses.append(np.std(np.array(valid_loss)))
    avg_accs.append(np.mean(np.array(valid_acc)))
    std_accs.append(np.std(np.array(valid_acc)))

plt.figure()
plt.errorbar(scales, avg_losses, yerr=std_losses)
plt.title("Mean Validation loss")
plt.xlabel("Scales")
plt.ylabel("Categorical cross entropy")
plt.savefig("valid_loss_cifar_scales.pdf")


plt.figure()
plt.errorbar(scales, avg_accs, yerr=std_losses)
plt.title("Mean Validation accuracy")
plt.xlabel("Scales")
plt.ylabel("Accuracy %")
plt.savefig("valid_acc_cifar_scales.pdf")

