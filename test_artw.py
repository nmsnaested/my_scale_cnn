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
from resNet import Model 

from artw_ds import GetArtDataset, EqSampler

from functions import train, test, plot_figures
import pickle
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=50
learning_rate = 0.00001
batch_size = 16
batch_log = 50

f_in=3
size=5
ratio=2**(2/3)
nratio=3
srange=2
padding=0
nb_classes=210

parameters = {
    "nb_epochs": nb_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "ratio": ratio,
    "nb_channels": nratio,
    "overlap": srange    
}
log = open("Artw_log.pickle", "wb")
pickle.dump(parameters, log)

train_transforms = torchvision.transforms.Compose([
                lambda path: Image.open(path),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.0), ratio=(1, 1)),
                torchvision.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

test_transforms = transforms.Compose([
                lambda path: Image.open(path), 
                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

base_dir = "./art_dataset"
train_set = GetArtDataset(basedir=base_dir, mode="train", transforms=train_transforms)
valid_set = GetArtDataset(basedir=base_dir, mode="val", transforms=test_transforms)
#test_set = GetArtDataset(basedir=base_dir, mode="test", transforms=test_transforms)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=1, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = True, num_workers=1, pin_memory=True)
#test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=4, pin_memory=True)

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

criterion = nn.CrossEntropyLoss()

models = [
    SiCNN_3(f_in, size, ratio, nratio, srange, padding, nb_classes),
    Model(f_in, size, ratio, nratio, srange, padding, nb_classes)
]

pickle.dump(len(models), log)

for model in models: 
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
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc
    }
    pickle.dump(dynamics, log)

log.close()

plot_figures("Artw_log.pickle", name="artwork", mode="train", mean = False)
plot_figures("Artw_log.pickle", name="artwork", mode="valid", mean = False)