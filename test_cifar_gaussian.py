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

from architectures import SiCNN_3, kanazawa

from functions import train, test, plot_figures
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=150
learning_rate = 0.00001
batch_size = 128
batch_log = 70
repeats = 6

f_in = 3
size = 5
ratio = 2**(2/3) 
nratio = 3
srange = 2
padding = 0

log = open("cifar_gaussian_log.pickle", "wb")

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "repetitions": 6,
    "size": size,
    "ratio": ratio,
    "nb channels": nratio
}
pickle.dump(parameters, log)

root = './cifardata'
if not os.path.exists(root):
    os.mkdir(root)

train_transf = transforms.Compose([
                transforms.Resize(40), RandomRescale(size = 40, scales = (1.0, 0.24), sampling = "normal"), 
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

train_set = datasets.CIFAR10(root=root, train=True, transform=train_transf, download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

criterion = nn.CrossEntropyLoss()

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

test_losses = []
test_accs = []

models = [
    kanazawa(f_in, ratio, nratio, srange=0),
    kanazawa(f_in, ratio, nratio, srange),
    SiCNN_3(f_in, size, ratio, nratio, srange=0),
    SiCNN_3(f_in, size, ratio, nratio, srange)
]
pickle.dump(len(models), log)

for model in models:
    model.to(device)

    for epoch in range(1, nb_epochs + 1): 
        train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
        train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
    
    #lists of last test loss and acc for each scale with model ii
    s_test_loss = [] 
    s_test_acc = []
    for s in scales: 
        test_transf = transforms.Compose([
                            transforms.Resize(40), RandomRescale(size = 40, scales = (s, s), sampling = "uniform"), 
                            transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        test_set = datasets.MNIST(root=root, train=False, transform=test_transf, download=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False, num_workers=1, pin_memory=True)

        test_l, test_a = test(model, test_loader, criterion, epoch, batch_log, device)

        s_test_loss.append(test_l) #take only last value 
        s_test_acc.append(test_a)

    results = {
        "model": model,
        "loss": s_test_loss,
        "acc": s_test_acc
    }
    pickle.dump(results, log)

log.close()

infile = open("cifar_gaussian_log.pickle", "rb")
params=pickle.load(infile)
nb_models=pickle.load(infile)
losses = []
accs = []
for ii in range(nb_models):
    res = pickle.load(infile)
    losses.append(res["loss"])
    accs.append(res["acc"])


plt.figure()
plt.plot(scales, losses[0], label="Kanazawa sr=0")
plt.plot(scales, losses[1], label="Kanazawa sr=2")
plt.plot(scales, losses[2], label="SiCNN_3 sr=0")
plt.plot(scales, losses[3], label="SiCNN_3 sr=2")
plt.title("Loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
plt.savefig("test_loss_gaussian_cifar.pdf")

plt.figure()
plt.plot(scales, accs[0], label="Kanazawa sr=0")
plt.plot(scales, accs[1], label="Kanazawa sr=2")
plt.plot(scales, accs[2], label="SiCNN_3 sr=0")
plt.plot(scales, accs[3], label="SiCNN_3 sr=2")
plt.title("Accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
plt.savefig("test_acc_gaussian_cifar.pdf")

err = [[100-x for x in l] for l in accs]

plt.figure()
plt.plot(scales, err[0], label="Kanazawa sr=0")
plt.plot(scales, err[1], label="Kanazawa sr=2")
plt.plot(scales, err[2], label="SiCNN_3 sr=0")
plt.plot(scales, err[3], label="SiCNN_3 sr=2")
plt.title("Error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.savefig("test_err_gaussian_cifar.pdf")