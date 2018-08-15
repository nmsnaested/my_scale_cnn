#pylint: disable=E1101

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

#import time_logging


def filter_size(size, ratio, nratio): 
    filter_size = math.ceil(size * ratio ** (nratio - 1))
    if filter_size % 2 != size % 2:
        filter_size += 1
    return filter_size


def train(model, train_loader, learning_rate, criterion, epoch, batch_log, device):
    running_loss = 0.0
    avg_loss = 0.0
    tot_acc = 0.0
    correct_cnt = 0
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model.train()

    for idx, (images,labels) in enumerate(train_loader, 0):
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        #t = time_logging.start()
        loss.backward()
        #time_logging.end("backward", t)
        optimizer.step()
        predicted = outputs.argmax(1)
        correct = (predicted == labels).long().sum().item()        
        correct_cnt += correct
        running_loss += loss.item()
        if idx % batch_log == (batch_log-1):  
            tot_acc = 100.*correct_cnt / (len(images)*(idx+1))
            avg_loss = running_loss/(idx+1)
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
                epoch, (idx+1)*len(images), len(train_loader.dataset), 100. * (idx+1) / len(train_loader), 
                loss.item(), avg_loss, tot_acc ))

    #print(time_logging.text_statistics())
    return avg_loss, tot_acc

def test(model,test_loader,criterion,epoch,batch_log,device):
    correct_cnt = 0
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for idx,(images,labels) in enumerate(test_loader):
            images,labels=images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(1)
            correct = (predicted == labels).long().sum().item()
            correct_cnt += correct
            
        avg_loss = total_loss / len(test_loader.dataset)
        tot_acc = 100. * correct_cnt / len(test_loader.dataset)
        print('Testing Epoch: {} \tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(epoch, avg_loss, tot_acc ))
        
    return avg_loss, tot_acc


def plot_figures(filename, name, mode, mean=False):
    '''
    :param mode: train, test or valid
    '''
    pickle_log = open(filename,'rb')
    params = pickle.load(pickle_log)
    nb_models = pickle.load(pickle_log)

    losses = []
    accs = []
    for ii in range(nb_models):
        dynamics = pickle.load(pickle_log)
        losses.append(dynamics["{}_loss".format(mode)])
        accs.append(dynamics["{}_acc".format(mode)])

    if mean:
        avg_loss = np.mean(np.array(losses), axis=0)
        avg_acc = np.mean(np.array(accs), axis=0)
        
        std_loss = np.std(np.array(losses), axis=0)
        std_acc = np.std(np.array(accs), axis=0)
        
        x = list(range(len(avg_loss)))

        plt.figure()
        plt.errorbar(x, avg_loss, yerr=std_loss)
        plt.title("Mean {} loss {}".format(mode, name)) 
        plt.xlabel("Epochs")
        plt.ylabel("Categorical cross entropy")
        plt.savefig("{}_loss_mean_{}.pdf".format(mode, name))

        plt.figure()
        plt.errorbar(x, avg_acc, yerr=std_acc)
        plt.title("Mean {} accuracy {}".format(mode, name))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("{}_acc_mean_{}.pdf".format(mode, name))

    else: 
        plt.figure()
        for ii in range(len(losses)):
            plt.plot(losses[ii], label = "model {}".format(ii))
        plt.title("{} loss {}".format(mode, name))
        plt.xlabel("Epochs")
        plt.ylabel("Categorical cross entropy")
        plt.legend()
        plt.savefig("{}_losses_{}.pdf".format(mode, name))
        #plt.show()

        plt.figure()
        for ii in range(len(accs)):
            plt.plot(accs[ii], label = "model {}".format(ii))
        plt.title("{} accuracy {}".format(mode, name))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("{}_accuracies_{}.pdf".format(mode, name))
        #plt.show()

    pickle_log.close()