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

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "filter size": size,
    "ratio": ratio,
    "nb channels": nratio,
    "overlap": srange    
}

criterion = nn.CrossEntropyLoss()

scales = [0.75, 0.875, 1.0] # crop center 24x24 or 28x28 and resize to 32x32, or leave 32x32 

log = open("cifar10_mean_log.pickle", "wb")
pickle.dump(parameters, log)

train_transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #mean 0 std 1 for RGB

idx = list(range(50000))
train_set = Subset(datasets.CIFAR10(root='./cifardata', train=True, transform=train_transf, download=True), idx[10000:])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#test_set = datasets.CIFAR10(root='./cifardata', train=False, transform=test_transf, download=True)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

pickle.dump(repeats, log)

s_avg_t_losses = []
s_std_t_losses = []
s_avg_t_accs = []
s_std_t_accs = []

s_avg_v_losses = []
s_std_v_losses = []
s_avg_v_accs = []
s_std_v_accs = []

for s in scales:
    test_transf = transforms.Compose([
    RandomRescale(size=32, sampling="uniform", scales=(s, s)), 
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    valid_set = Subset(datasets.CIFAR10(root='./cifardata', train=True, transform=test_transf, download=True), idx[:10000])
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    m_t_losses = []
    m_t_accs = []
    m_v_losses = []
    m_v_accs = []

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
            valid_l, valid_a = test(model, valid_loader, criterion, epoch, batch_log, device)
            train_loss.append(train_l)
            train_acc.append(train_a)
            valid_loss.append(valid_l)
            valid_acc.append (valid_a)
        
        with open("model_{}_{}_cifar.pickle".format(s, ii), "wb") as save:
            pickle.dump(model, save)
    
        m_t_losses.append(train_loss)
        m_t_accs.append(train_acc)
        m_v_losses.append(valid_loss)
        m_v_losses.append(valid_acc)

    s_avg_t_losses.append(np.mean(np.array(m_t_losses)))
    s_std_t_losses.append(np.std(np.array(m_t_losses)))
    s_avg_t_accs.append(np.mean(np.array(m_t_accs)))
    s_std_t_accs.append(np.std(np.array(m_t_accs)))

    s_avg_v_losses.append(np.mean(np.array(m_v_losses)))
    s_std_v_losses.append(np.std(np.array(m_v_losses)))
    s_avg_v_accs.append(np.mean(np.array(m_v_accs)))
    s_std_v_accs.append(np.std(np.array(m_v_accs)))

pickle.dump(s_avg_t_losses, log)
pickle.dump(s_avg_t_accs, log)
pickle.dump(s_std_t_losses, log)
pickle.dump(s_std_t_accs, log)

pickle.dump(s_avg_v_losses, log)
pickle.dump(s_avg_v_accs, log)
pickle.dump(s_std_v_losses, log)
pickle.dump(s_std_v_accs, log)

log.close()

plt.figure()
for i, s in enumerate(scales):
    plt.errorbar(list(range(len(s_avg_t_losses[i]))), s_avg_t_losses[i], yerr=s_std_t_losses[i], label = "scale {}".format(s))
plt.title("Average train loss")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.savefig("train_loss_cifar_scales.pdf")

plt.figure()
for i, s in enumerate(scales):
    plt.errorbar(list(range(len(s_avg_t_accs[i]))), s_avg_t_accs[i], yerr=s_std_t_accs[i], label = "scale {}".format(s))
plt.title("Average train accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.savefig("train_acc_cifar_scales.pdf")

plt.figure()
for i, s in enumerate(scales):
    plt.errorbar(list(range(len(s_avg_v_losses[i]))), s_avg_v_losses[i], yerr=s_std_v_losses[i], label = "scale {}".format(s))
plt.title("Average validation loss")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.savefig("valid_loss_cifar_scales.pdf")

plt.figure()
for i, s in enumerate(scales):
    plt.errorbar(list(range(len(s_avg_v_accs[i]))), s_avg_v_accs[i], yerr=s_std_v_accs[i], label = "scale {}".format(s))
plt.title("Average validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.savefig("valid_acc_cifar_scales.pdf")