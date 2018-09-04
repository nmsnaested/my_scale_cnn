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

nb_epochs=150
learning_rate = 0.0001
batch_size = 128
batch_log = 70
repeats = 6

f_in = 1
size=5
ratio=2**(2/3)
nratio=3
srange=0
padding=0

log = open("mnist_range_sr0_log.pickle", "wb")

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

scales = [(1.0, 1.0), (0.9, 1.1), (0.8, 1.2), (0.6, 1.4), (0.5, 1.5), (0.4, 1.6), (0.3, 1.7)]

pickle.dump(scales, log)

criterion = nn.CrossEntropyLoss()

avg_test_losses = []
avg_test_accs = []
std_test_losses = []
std_test_accs = []

for scale in scales:
    uniform = transforms.Compose([
                transforms.Resize(40), RandomRescale(size = 40, scales = scale, sampling = "uniform"), 
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

    root = './mnistdata'
    if not os.path.exists(root):
        os.mkdir(root)

    train_set = datasets.MNIST(root=root, train=True, transform=uniform, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    test_set = datasets.MNIST(root=root, train=False, transform=uniform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    s_test_losses = []
    s_test_accs = []

    for ii in range(repeats):
        model = SiCNN_3(f_in, size, ratio, nratio, srange, padding)
        model.to(device)

        for epoch in range(1, nb_epochs + 1): 
            train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
            train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 

        test_l, test_a = test(model, test_loader, criterion, epoch, batch_log, device)
            
        s_test_losses.append(test_l)
        s_test_accs.append(test_a)

        pickle.dump(model, log)

    dynamics = {
        "scale": scale,
        "test_losses": s_test_losses,
        "test_accs": s_test_accs
    }
    pickle.dump(dynamics, log)
    
    mean_l = np.mean(np.array(s_test_losses))
    std_l = np.std(np.array(s_test_losses))
    mean_a = np.mean(np.array(s_test_accs))
    std_a = np.std(np.array(s_test_accs))

    avg_test_losses.append(mean_l)
    avg_test_accs.append(mean_a)
    std_test_losses.append(std_l)
    std_test_accs.append(std_a)

pickle.dump(avg_test_losses, log)
pickle.dump(std_test_losses, log)
pickle.dump(avg_test_accs, log)
pickle.dump(std_test_accs, log)

log.close()

plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_losses, yerr=std_test_losses)
plt.title("Average loss vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_range_mean_sr0.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_accs, yerr=std_test_accs)
plt.title("Average accuracy vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("test_acc_range_mean_sr0.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs], yerr=std_test_accs)
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("test_err_range_mean_sr0.pdf")