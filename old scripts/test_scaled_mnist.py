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
import loaddataset as lds

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from mnistCNNs import SiCNN, Model

from functions import train, test
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=100
learning_rate = 0.0001
batch_size = 128
batch_log = 50


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
rescale = transforms.Compose([RandomRescale(size = 28, scale = (0.3, 1)), 
                            transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

root = './mnistdata'
if not os.path.exists(root):
    os.mkdir(root)
train_set = datasets.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)

test_set = datasets.MNIST(root=root, train=False, transform=rescale, download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False)

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

sicnn = SiCNN(2**(2/3), 3)
sicnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sicnn.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

sicnn_train_loss=[]
sicnn_train_acc = []
sicnn_test_loss = []
sicnn_test_acc = []
sicnn_dyn = []

for epoch in range(1, nb_epochs + 1): 
    scheduler.step() 
    train_l, train_a = train(sicnn, train_loader, optimizer, criterion, epoch, batch_log, device) 
    train_l, train_a = test(sicnn, train_loader, criterion, epoch, batch_log, device) 
    test_l, test_a = test(sicnn, test_loader, criterion, epoch, batch_log, device)
    sicnn_train_loss.append(train_l)
    sicnn_train_acc.append(train_a)
    sicnn_test_loss.append(test_l)
    sicnn_test_acc.append(test_a)

with open("mnist_sicnn_log_decaylr.txt", "w") as output:
    output.write("nb_epochs=50\t lr=0.001 w/ decay every 20 epochs\t batch_size=128\n")
    output.write("SiCNN \t ratio=2^(2/3), nratio=3")
    output.write(str(sicnn_train_loss))
    output.write(str(sicnn_train_acc))
    output.write(str(sicnn_test_loss))
    output.write(str(sicnn_test_acc))

plt.figure()
plt.plot(sicnn_train_loss, label = "SiCNN")
plt.title("Training loss MNIST")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_mnist.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_train_acc, label = "SiCNN")
plt.title("Training accuracy MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_mnist.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_test_loss, label = "SiCNN")
plt.title("Test loss Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_mnist_scaled.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_test_acc, label = "SiCNN")
plt.title("Test accuracy Scaled MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_mnist_scaled.pdf")
