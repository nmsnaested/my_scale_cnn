#pylint: disable=E1101
import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

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

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]


"""
infile = open("mnist_range_log.pickle", "rb")

params = pickle.load(infile)
repeats = params["repetitions"]
batch_size = params["batch size"]
scales = pickle.load(infile)
criterion = nn.CrossEntropyLoss()

avg_test_losses = []
avg_test_accs = []
std_test_losses = []
std_test_accs = []
for scale in scales:
    uniform = transforms.Compose([
                transforms.Resize(40), RandomRescale(size = 40, scales = scale, sampling = "uniform"), 
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

    test_set = datasets.MNIST(root='./mnistdata', train=False, transform=uniform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    s_test_losses = []
    s_test_accs = []

    for ii in range(repeats):
        model = pickle.load(infile)
        dynamics = pickle.load(infile)
        
        test_l, test_a = test(model, test_loader, criterion, 200, 70, device)

        s_test_losses.append(test_l)
        s_test_accs.append(test_a)
    
    mean_l = np.mean(np.array(s_test_losses))
    std_l = np.std(np.array(s_test_losses))
    mean_a = np.mean(np.array(s_test_accs))
    std_a = np.std(np.array(s_test_accs))

    avg_test_losses.append(mean_l)
    avg_test_accs.append(mean_a)
    std_test_losses.append(std_l)
    std_test_accs.append(std_a)

infile.close()


plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_losses, yerr=std_test_losses)
plt.title("Average loss vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_range_mean.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_accs, yerr=std_test_accs)
plt.title("Average accuracy vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("test_acc_range_mean.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs], yerr=std_test_accs)
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("test_err_range_mean.pdf")

"""
lists = []
infile = open('mnist_gaussian_log.pickle', 'rb')
while 1:
    try:
        lists.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs = lists[-1]
std_test_losses = lists[-2]
avg_test_accs = lists[-3]
avg_test_losses = lists[-4]

lists_sr0 = []
infile = open('mnist_gaussian_sr0_log.pickle', 'rb')
while 1:
    try:
        lists_sr0.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs_sr0 = lists_sr0[-1]
std_test_losses_sr0 = lists_sr0[-2]
avg_test_accs_sr0 = lists_sr0[-3]
avg_test_losses_sr0 = lists_sr0[-4]

kanazawa = [9.082670906200317, 5.104928457869634, 2.7726550079491243, 1.8139904610492845, 1.7853736089030203, 1.4419713831478518, 1.585055643879171, 1.5707472178060407, 2.0715421303656587, 3.0731319554848966, 4.103338632750397]
convnet = [11.286168521462638, 6.449920508744037, 3.4737678855325917, 2.3147853736089026, 2.114467408585055, 1.742448330683624, 1.8426073131955487, 2.128775834658187, 3.1589825119236874, 5.319554848966613, 7.666136724960254]

plt.figure()
plt.errorbar(scales, avg_test_losses, yerr=std_test_losses, label="srange=2")
plt.errorbar(scales, avg_test_losses_sr0, yerr=std_test_losses_sr0, label="srange=0")
plt.title("Average loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_gaussian_mean_all.pgf")

plt.figure()
plt.errorbar(scales, avg_test_accs, yerr=std_test_accs, label="srange=2")
plt.errorbar(scales, avg_test_accs_sr0, yerr=std_test_accs_sr0, label="srange=0")
plt.title("Average accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("test_acc_gaussian_mean_all.pgf")

plt.figure()
plt.errorbar(scales, [100-x for x in avg_test_accs], yerr=std_test_accs, label="srange=2")
plt.errorbar(scales, [100-x for x in avg_test_accs_sr0], yerr=std_test_accs_sr0, label="srange=0")
#plt.errorbar(scales, kanazawa, label="Kanazawa")
#plt.errorbar(scales, convnet, label="ConvNet")
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("test_err_gaussian_mean_all.pgf")

