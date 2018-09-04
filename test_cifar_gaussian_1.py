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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from architectures import SiCNN_3, kanazawa, SiCNN_3big

from functions import filter_size, train, test, plot_figures
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=200
learning_rate = 0.00001
batch_size = 128
batch_log = 70
repeats = 4

f_in = 3
size = 5
ratio = 2**(2/3) 
nratio = 3
srange = 2
padding = 0

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "repetitions": repeats,
    "size": size,
    "ratio": ratio,
    "nb channels": nratio
}

log = open("cifar_gaussian_log_results.pickle", "wb")

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

root = './cifardata' 
if not os.path.exists(root):
    os.mkdir(root)

criterion = nn.CrossEntropyLoss()

models = [
        #kanazawa(f_in, ratio, nratio, srange=0),
        #kanazawa(f_in, ratio, nratio, srange),
        SiCNN_3(f_in, size, ratio, nratio, srange=0),
        SiCNN_3(f_in, size, ratio, nratio, srange),
        SiCNN_3(f_in, filter_size(size, ratio, nratio), 1/ratio, nratio, srange),
        SiCNN_3big(f_in, size, ratio, nratio, srange=0), 
        SiCNN_3(f_in, size, ratio=2**(1/3), nratio=6, srange=0), 
        SiCNN_3(f_in, size, ratio=2**(1/3), nratio=6, srange=srange), 
        SiCNN_3(f_in, filter_size(size, 2**(1/3), 6), ratio=2**(-1/3), nratio=6, srange=srange)
    ]

for m in range(len(models)):
    locals()['test_losses_{0}'.format(m)] = []
    locals()['test_accs_{0}'.format(m)] = []

for ii in range(repeats):
    for m in range(len(models)):

        model = pickle.load(open("./trained_models/trained_model_{}_{}.pickle".format(m, ii), "rb"))
        model.to(device)

        #lists of last test loss and acc for each scale with model m, trial ii
        s_test_loss = [] 
        s_test_acc = []
        for s in scales: 
            test_transf = transforms.Compose([
                                transforms.Resize(40), RandomRescale(size = 40, scales = (s, s), sampling = "uniform"), 
                                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
            test_set = datasets.CIFAR10(root=root, train=False, transform=test_transf, download=True)
            test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False, num_workers=1, pin_memory=True)

            test_l, test_a = test(model, test_loader, criterion, 200, batch_log, device)

            s_test_loss.append(test_l) #take only last value 
            s_test_acc.append(test_a)
        
        locals()['test_losses_{0}'.format(m)].append(s_test_loss)
        locals()['test_accs_{0}'.format(m)].append(s_test_acc)

for m in range(len(models)):
    pickle.dump(locals()['test_losses_{0}'.format(m)], log)
    pickle.dump(locals()['test_accs_{0}'.format(m)], log)


for m in range(len(models)):
    locals()['avg_test_loss_{0}'.format(m)] = np.mean(np.array(locals()['test_losses_{0}'.format(m)]), axis=0)
    locals()['avg_test_acc_{0}'.format(m)] = np.mean(np.array(locals()['test_accs_{0}'.format(m)]), axis=0)
    locals()['std_test_loss_{0}'.format(m)] = np.std(np.array(locals()['test_losses_{0}'.format(m)]), axis=0)
    locals()['std_test_acc_{0}'.format(m)] = np.std(np.array(locals()['test_accs_{0}'.format(m)]), axis=0)

    pickle.dump(locals()['avg_test_loss_{0}'.format(m)], log)
    pickle.dump(locals()['avg_test_acc_{0}'.format(m)], log)
    pickle.dump(locals()['std_test_loss_{0}'.format(m)], log)
    pickle.dump(locals()['std_test_acc_{0}'.format(m)], log)

log.close()

