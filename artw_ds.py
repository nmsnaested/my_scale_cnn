import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import random
import matplotlib.pyplot as plt
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



class GetArtDataset(Dataset):
    def __init__(self, basedir, mode, transforms=None):
        '''
        :param basedir: name of base directory in which folders of dataset can be found
        :param mode: train, val or test 
        :param transforms: list of transformations to be applied on the image 
        '''
        self.imgdir = os.path.join(basedir, "images")
        self.labdir = os.path.join(basedir, "labels")
        self.transforms = transforms
        self.mode = mode

        self.labels = []
        self.images = []
        self.names = []
        
        file = os.path.join(self.labdir, "{}.txt".format(self.mode))
        with open(file, "rt") as f:
            for line in f.readlines():
                self.images.append(line.split(' ')[0])
                self.labels.append(line.split(' ')[1])
        
        file = os.path.join(self.labdir, "names.txt")
        with open(file, "rt") as f:
            for line in f.readlines():
                self.names.append(line.split("\t")[0])

    def __getitem__(self,idx):
        name = self.images[idx]
        image = os.path.join(self.imgdir, name)
        if self.transforms is not None:
            image = self.transforms(image)
        label = int(self.labels[idx])
        return image, label

    def __len__(self):
        return len(self.images)

