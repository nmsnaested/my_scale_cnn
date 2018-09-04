#pylint: disable=E1101

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from loaddataset import ImgNetDataset

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class SiCNN(nn.Module): 
    def __init__(self, f_in=3, size=5, ratio=2**(2/3), nratio=3, srange=2, padding=0, nb_classes=10): 
        super().__init__()
        '''
        Scale equivariant arch with 3 convolutional layers
        '''
        self.f_in = f_in
        self.size = size
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange
        self.padding = padding
        self.nb_classes = nb_classes

        self.conv1 = ScaleConvolution(self.f_in, 96, self.size, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", padding=self.padding, stride = 2)
        self.conv2 = ScaleConvolution(96, 256, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv3 = ScaleConvolution(256, 384, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(384, 1024, bias=True)
        self.fc2 = nn.Linear(1024, self.nb_classes, bias=True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

nb_epochs=1
learning_rate = 0.001
batch_size = 128

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

root = "./tiny-imagenet-200"
train_set = ImgNetDataset(rootdir=root, mode="train", transforms=transforms)
valid_set = ImgNetDataset(rootdir=root, mode="val", transforms=transforms)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=1, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True, num_workers=1, pin_memory=True)

model = SiCNN(f_in=3, size=5, ratio=2**(2/3), nratio=3, srange=2, padding=0, nb_classes=200)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(nb_epochs):

    print("train")
    model.train()

    total_loss = 0.0
    correct_cnt = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predicted = outputs.argmax(1)
        correct = (predicted == labels).long().sum().item()        
        correct_cnt += correct
        total_loss += loss.item()

    print("train bis")
    model.eval()

    total_loss = 0.0
    correct_cnt = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(1)
            correct = (predicted == labels).long().sum().item()
            correct_cnt += correct
            
        test_loss = total_loss / len(valid_loader.dataset)
        test_acc = 100. * correct_cnt / len(valid_loader.dataset)
        print('Training Epoch: {} \tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
            epoch, test_loss, test_acc ))

    print("validation")
    total_loss = 0.0
    correct_cnt = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(1)
            correct = (predicted == labels).long().sum().item()
            correct_cnt += correct
            
        test_loss = total_loss / len(valid_loader.dataset)
        test_acc = 100. * correct_cnt / len(valid_loader.dataset)
        print('Validation Epoch: {} \tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
            epoch, test_loss, test_acc ))

"""
test_set = ImgNetDataset(rootdir=root, mode="test", transforms=transforms)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=4, pin_memory=True)


total_loss = 0.0
correct_cnt = 0.0
with torch.no_grad():
    for idx, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        outputs = model(images)
        pred_label = outputs.argmax(1)
        pred_class = test_set.names[pred_label].cpu().numpy() #copy tensor back to CPU and convert to numpy
        np.savetxt('predictions.txt', pred_class)


"""