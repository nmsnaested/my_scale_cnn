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
batch_size = 256
batch_log = 70
repeats = 6

f_in = 1
size = 5
ratio = 2**(2/3) 
nratio = 3
srange = 2 
padding = 0

log = open("mnist_gaussian_log.pickle", "wb")

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "repetitions": 6,
    "size": size,
    "ratio": ratio,
    "nb channels": nratio,
    "overlap": srange
}
pickle.dump(parameters, log)

root = './mnistdata'
if not os.path.exists(root):
    os.mkdir(root)

train_transf = transforms.Compose([
                transforms.Resize(40), RandomRescale(size = 40, scales = (1.0, 0.24), sampling = "normal"), 
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

train_set = datasets.MNIST(root=root, train=True, transform=train_transf, download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

criterion = nn.CrossEntropyLoss()

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

pickle.dump(len(scales), log)

test_losses = []
test_accs = []

for ii in range(repeats):
    model = SiCNN_3(f_in, size, ratio, nratio, srange, padding)
    model.to(device)

    for epoch in range(1, nb_epochs + 1): 
        train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
        train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
    
    pickle.dump(model, open("model_{}_trained.pickle".format(ii), "wb"))

    #lists of last test loss and acc for each scale with model ii
    s_test_loss = [] 
    s_test_acc = []
    for s in scales: 
        test_transf = transforms.Compose([
                            transforms.Resize(40), RandomRescale(size = 40, scales = (s, s), sampling = "uniform"), 
                            transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        test_set = datasets.MNIST(root=root, train=False, transform=test_transf, download=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False)

        for epoch in range(1, nb_epochs + 1):  
            test_l, test_a = test(model, test_loader, criterion, epoch, batch_log, device)

        s_test_loss.append(test_l) #take only last value 
        s_test_acc.append(test_a)

    results = {
        "trial": ii,
        "losses": s_test_loss,
        "accs": s_test_acc
    }
    pickle.dump(results, log)

    # lists of lists w/ all test loss and accs, each column is a given scale
    test_losses.append(s_test_loss) 
    test_accs.append(s_test_acc)

avg_test_losses = np.mean(np.array(test_losses), axis=0) #averaging over the columns to get mean value for each scale
avg_test_accs = np.mean(np.array(test_accs), axis=0)

std_test_losses = np.std(np.array(test_losses), axis=0)
std_test_accs = np.std(np.array(test_accs), axis=0)

pickle.dump(avg_test_losses, log)
pickle.dump(avg_test_accs, log)
pickle.dump(std_test_losses, log)
pickle.dump(std_test_accs, log)

log.close()

plt.figure()
plt.plot(s, avg_test_losses, yerr=std_test_losses)
plt.title("Average loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_gaussian_mean.pdf")

plt.figure()
plt.plot(s, avg_test_accs, yerr=std_test_accs)
plt.title("Average accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("test_acc_gaussian_mean.pdf")

plt.figure()
plt.plot(s, [100-x for x in avg_test_accs], yerr=std_test_accs)
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("test_err_gaussian_mean.pdf")