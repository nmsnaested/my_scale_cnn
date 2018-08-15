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

import artw_ds
from artw_ds import GetArtDataset, EqSampler

from functions import train, test, plot_figures
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=50
learning_rate = 0.00001
batch_size = 32
batch_log = 50

ratio = (2**(1/3))
nratio = 8 #see comparisons for parameter optimisation

parameters = {
    "nb_epochs": nb_epochs,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "ratio": ratio,
    "nb_channels": nratio    
}
phandle = open("Artw_log.pickle", "wb")
pickle.dump(parameters, phandle)

train_transforms = torchvision.transforms.Compose([
                lambda path: PIL.Image.open(path),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.0), ratio=(1, 1)),
                torchvision.transforms.ToTensor(),
            ])

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

base_dir = "./art_dataset"
train_set = artw_ds.GetArtDataset(basedir=base_dir, mode="train", transforms=train_transforms)

test_set = artw_ds.GetArtDataset(basedir=base_dir, mode="test", transforms=test_transforms)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=4, pin_memory=True)


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

sicnn = SiCNN(ratio, nratio)
sicnn.to(device)

resnet = Model(size=5,ratio=ratio,nratio=nratio,srange=2,padding=2)
resnet.to(device)

allcnn = SiAllCNN(ratio, nratio)
allcnn.to(device)

criterion = nn.CrossEntropyLoss()
sicnn_optimizer = optim.Adam(sicnn.parameters(), lr = learning_rate)
res_optimizer = optim.Adam(resnet.parameters(), lr = learning_rate)
all_optimizer = optim.Adam(allcnn.parameters(), lr = learning_rate)


sicnn_train_loss=[]
sicnn_train_acc = []
sicnn_test_loss = []
sicnn_test_acc = []

sicnn_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(sicnn, train_loader, sicnn_optimizer, criterion, epoch, batch_log, device) 
    test_l, test_a = test(sicnn, test_loader, criterion, epoch, batch_log, device)
    sicnn_train_loss.append(train_l)
    sicnn_train_acc.append(train_a)
    sicnn_test_loss.append(test_l)
    sicnn_test_acc.append(test_a)

    sicnn_dyn.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

with open("SiCNN_artw_log.txt", "w") as output:
    output.write(str(sicnn_train_loss))
    output.write(str(sicnn_train_acc))
    output.write(str(sicnn_test_loss))
    output.write(str(sicnn_test_acc))


plt.figure()
plt.plot(sicnn_train_loss, label = "SiCNN")
plt.title("Training loss TICC Printmaking Dataset")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_art.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_train_acc, label = "SiCNN")
plt.title("Training accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_art.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_test_loss, label = "SiCNN")
plt.title("Test loss TICC Printmaking Dataset") 
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_art.pdf")
#plt.show()

plt.figure()
plt.plot(sicnn_test_acc, label = "SiCNN")
plt.title("Test accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_art.pdf")
#plt.show()


all_train_loss=[]
all_train_acc = []
all_test_loss = []
all_test_acc = []

all_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(allcnn, train_loader, all_optimizer, criterion, epoch, batch_log, device) 
    train_l, train_a = test(allcnn, train_loader, criterion, epoch, batch_log, device) 
    all_train_loss.append(train_l)
    all_train_acc.append(train_a)

    test_l, test_a = test(allcnn, test_loader, criterion, epoch, batch_log, device)
    all_test_loss.append(test_l)
    all_test_acc.append(test_a)

    all_dyn.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

with open("SiAllCNN_artw_log.txt", "w") as output:
    output.write(str(all_train_loss))
    output.write(str(all_train_acc))
    output.write(str(all_test_loss))
    output.write(str(all_test_acc))


plt.figure()
plt.plot(all_train_loss, label = "Si-AllCNN")
plt.title("Training loss TICC Printmaking Dataset")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_art_all.pdf")
#plt.show()

plt.figure()
plt.plot(all_train_acc, label = "Si-AllCNN")
plt.title("Training accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_art_all.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_loss, label = "Si-AllCNN")
plt.title("Test loss TICC Printmaking Dataset") 
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_art_all.pdf")
#plt.show()

plt.figure()
plt.plot(all_test_acc, label = "Si-AllCNN")
plt.title("Test accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_art_all.pdf")
#plt.show()

res_train_loss=[]
res_train_acc = []
res_test_loss = []
res_test_acc = []

res_dyn = []

for epoch in range(1, nb_epochs + 1):  
    train_l, train_a = train(resnet, train_loader, res_optimizer, criterion, epoch, batch_log, device) 
    train_l, train_a = test(resnet, train_loader, criterion, epoch, batch_log, device) 
    res_train_loss.append(train_l)
    res_train_acc.append(train_a)

    test_l, test_a = test(resnet, test_loader, criterion, epoch, batch_log, device)
    res_test_loss.append(test_l)
    res_test_acc.append(test_a)

    res_dyn.append({
        "epoch": epoch,
        "train_loss": train_l,
        "train_acc": train_a,
        "test_loss": test_l,
        "test_acc": test_a
    })

with open("ResNet_artw_log.txt", "w") as output:
    output.write(str(res_train_loss))
    output.write(str(res_train_acc))
    output.write(str(res_test_loss))
    output.write(str(res_test_acc))

pickle.dump({"sicnn": sicnn_dyn, "allcnn": all_dyn, "resnet": res_dyn}, phandle)
phandle.close()

plt.figure()
plt.plot(res_train_loss, label = "ResNet")
plt.title("Training loss TICC Printmaking Dataset")
plt.xlabel("Epochs")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("training_loss_art_res.pdf")
#plt.show()

plt.figure()
plt.plot(res_train_acc, label = "ResNet")
plt.title("Training accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_acc_art_res.pdf")
#plt.show()

plt.figure()
plt.plot(res_test_loss, label = "ResNet")
plt.title("Test loss TICC Printmaking Dataset") 
plt.xlabel("Epoch")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_art_res.pdf")
#plt.show()

plt.figure()
plt.plot(res_test_acc, label = "ResNet")
plt.title("Test accuracy TICC Printmaking Dataset")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("test_acc_art_res.pdf")
#plt.show()