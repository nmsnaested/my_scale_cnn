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

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class StdNet(nn.Module):
    def __init__(self):
        super(StdNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 7,2)
        self.conv2 = nn.Conv2d(12, 21, 5)
        self.fc1 = nn.Linear(21*9*9, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 21*9*9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiCNN(nn.Module): 
    def __init__(self,ratio,nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(7,3,12, self.ratio,self.nratio,srange=0,boundary_condition="neumann",stride=2)
        self.conv2 = ScaleConvolution(5,12,21, self.ratio,self.nratio,srange=1,boundary_condition="neumann")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(21, 150, bias=True)
        self.fc2 = nn.Linear(150, 10, bias=True)

    def forward(self, x): 
        # the 2 following lines are scale+translation equivariant
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiCNN_d(nn.Module): 
    def __init__(self,ratio,nratio): 
        super().__init__()
        
        self.ratio = ratio 
        self.nratio = nratio

        self.conv1 = ScaleConvolution(7,3,12, self.ratio,self.nratio,srange=0,boundary_condition="dirichlet",stride=2)
        self.conv2 = ScaleConvolution(5,12,21, self.ratio,self.nratio,srange=1,boundary_condition="dirichlet")
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(21, 150, bias=True)
        self.fc2 = nn.Linear(150, 10, bias=True)

    def forward(self, x): 
        # the 2 following lines are scale+translation equivariant
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

nb_epochs=50
learning_rate = 0.0001
batch_size = 128
batch_log = 50

ratio = 2**(1/3)
nratio = 6

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #normalization for RGB
train_set = datasets.CIFAR10(root='./cifardata',train=True,transform=transform,download=True)

test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=transform,download=True)

#resize = transforms.Compose([transforms.RandomResizedCrop(size=128,scale=(0.05,1),ratio=(1,1)),transforms.Resize(size=32),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#test_set = datasets.CIFAR10(root='./cifardata',train=False,transform=resize,download=True)

train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

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
stdnet = StdNet()
stdnet.to(device)

sicnn = SiCNN(ratio,nratio)
sicnn.to(device)

criterion = nn.CrossEntropyLoss()
std_optimizer = optim.Adam(stdnet.parameters(),lr=learning_rate)
sicnn_optimizer = optim.Adam(sicnn.parameters(),lr=learning_rate)

def train(model,train_loader,optimizer,criterion,epoch,batch_log):
    running_loss = 0.0
    avg_loss = 0.0
    tot_acc=0.0
    correct_cnt = 0
    for idx, (images,labels) in enumerate(train_loader, 0):
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()        
        correct_cnt += correct
        running_loss += loss.item()
        if idx % batch_log == (batch_log-1):  
            tot_acc = 100.*correct_cnt / (len(images)*(idx+1))
            avg_loss = running_loss/(idx+1)
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
                epoch, (idx+1)*len(images), len(train_loader.dataset), 100. * (idx+1) / len(train_loader), 
                loss.item(), avg_loss, tot_acc ))
    return avg_loss, tot_acc

def test(model,test_loader,criterion,epoch,batch_log):
    correct_cnt = 0
    total_loss = 0.0
    with torch.no_grad():
        for idx,(images,labels) in enumerate(test_loader):
            images,labels=images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            correct_cnt += correct
            
        avg_loss = total_loss / len(test_loader.dataset)
        tot_acc = 100. * correct_cnt / len(test_loader.dataset)
        print('Testing Epoch: {} \tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(epoch, avg_loss, tot_acc ))
        
    return avg_loss, tot_acc

std_train_loss=[]
std_train_acc = []
std_test_loss = []
std_test_acc = []

for epoch in range(1,nb_epochs+1):  
    train_l,train_a = train(stdnet,train_loader,std_optimizer,criterion,epoch,batch_log) 
    test_l, test_a = test(stdnet,test_loader,criterion,epoch,batch_log)
    std_train_loss.append(train_l)
    std_train_acc.append(train_a)
    std_test_loss.append(test_l)
    std_test_acc.append(test_a)

with open("std_log_0.0001.txt", "w") as output:
    output.write(str(std_train_loss))
    output.write(str(std_train_acc))
    output.write(str(std_test_loss))
    output.write(str(std_test_acc))

sicnn_train_loss=[]
sicnn_train_acc = []
sicnn_test_loss = []
sicnn_test_acc = []

for epoch in range(1,nb_epochs+1):  
    train_l,train_a = train(sicnn,train_loader,sicnn_optimizer,criterion,epoch,batch_log) 
    test_l, test_a = test(sicnn,test_loader,criterion,epoch,batch_log)
    sicnn_train_loss.append(train_l)
    sicnn_train_acc.append(train_a)
    sicnn_test_loss.append(test_l)
    sicnn_test_acc.append(test_a)

with open("sicnn_log_0.0001.txt", "w") as output:
    output.write(str(sicnn_train_loss))
    output.write(str(sicnn_train_acc))

    output.write(str(sicnn_test_loss))
    output.write(str(sicnn_test_acc))

plt.figure()
plt.plot(std_train_loss, label = "Standard ConvNet")
plt.plot(sicnn_train_loss, label = "SiCNN")
plt.title("Training loss CIFAR10 lr=0.0001")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_cifar10_lr=0.0001.pdf")
#plt.show()

plt.figure()
plt.plot(std_train_acc, label = "Standard ConvNet")
plt.plot(sicnn_train_acc, label = "SiCNN")
plt.title("Training accuracy CIFAR10 lr=0.0001")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_cifar10_lr=0.0001.pdf")
#plt.show()

plt.figure()
plt.plot(std_test_loss, label = "Standard ConvNet")
plt.plot(sicnn_test_loss, label = "SiCNN")
plt.title("Test loss CIFAR10 lr=0.0001")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_cifar10_lr=0.0001.pdf")
#plt.show()

plt.figure()
plt.plot(std_test_acc, label = "Standard ConvNet")
plt.plot(sicnn_test_acc, label = "SiCNN")
plt.title("Test accuracy CIFAR10 lr=0.0001")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_cifar10_lr=0.0001.pdf")
#plt.show()
