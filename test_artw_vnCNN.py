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
import loaddataset as lds
from loaddataset import GetArtDataset

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from artwCNNs import SiVN

from functions import train, test
from rescale import RandomResizedCrop
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=50
learning_rate = 0.0001
batch_size = 32
batch_log = 50

ratio = 2**(1/3)
nratio = 8 #see comparisons for parameter optimisation

train_transforms = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transforms = transforms.Compose([transforms.Resize(512), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

base_dir = "./art_dataset"
train_set = lds.GetArtDataset(basedir=base_dir, transforms=train_transforms, train=True)
test_set = lds.GetArtDataset(basedir=base_dir, transforms=test_transforms, train=False)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=4, pin_memory=True)

print("Datasets done")

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

sivn = SiVN(ratio, nratio)
sivn.to(device)

criterion = nn.CrossEntropyLoss()
vn_optimizer = optim.Adam(sivn.parameters(), lr = learning_rate)

vn_train_loss=[]
vn_train_acc = []
vn_test_loss = []
vn_test_acc = []

vn_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(sivn, train_loader, vn_optimizer, criterion, epoch, batch_log, device) 
    train_l, train_a = test(sivn, train_loader, criterion, epoch, batch_log, device) 
    vn_train_loss.append(train_l)
    vn_train_acc.append(train_a)

    test_l, test_a = test(sivn, test_loader, criterion, epoch, batch_log, device)
    vn_test_loss.append(test_l)
    vn_test_acc.append(test_a)

with open("siVN_log.txt", "w") as output:
    output.write("parameters:\t nb_epochs=100\tlearning_rate=0.00001\tbatch_size=128\tratio=2^(1/3)\tnb channels(nratio)=8 \n")
    output.write(str(vn_train_loss))
    output.write(str(vn_train_acc))
    output.write(str(vn_test_loss))
    output.write(str(vn_test_acc))

plt.figure()
plt.plot(vn_train_loss, label = "Si-VanNoord CNN")
plt.title("Training loss TICC Printmaking Dataset")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_art_VN.pdf")
#plt.show()

plt.figure()
plt.plot(vn_train_acc, label = "Si-VanNoord CNN")
plt.title("Training accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_art_VN.pdf")
#plt.show()

plt.figure()
plt.plot(vn_test_loss, label = "Si-VanNoord CNN")
plt.title("Test loss TICC Printmaking Dataset") 
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_art_VN.pdf")
#plt.show()

plt.figure()
plt.plot(vn_test_acc, label = "Si-VanNoord CNN")
plt.title("Test accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_art_VN.pdf")
#plt.show()