#pylint: disable=E1101
import os, sys
import os.path
import gzip
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class DatasetFromDir(Dataset):
    def __init__(self, dirname, transforms=None):
        self.dirname = dirname
        self.list = [filename for filename in os.listdir(self.dirname) if os.path.isfile(os.path.join(self.dirname, filename)) and filename!='.DS_Store']
        self.transforms = transforms

    def __getitem__(self, idx):
        filename = self.list[idx]
        f = gzip.GzipFile(os.path.join(self.dirname, filename), "r")
        tmp = filename.split('|') #ex: filename = 265|C=5|S=1|T=(0, 0)
        img = np.load(f)
        img = (torch.from_numpy(img)).unsqueeze(0)
        img = self.transforms(img)
        label = int(list(tmp[1])[2])
        return img, label 

    def __len__(self):    
        return len(self.list)

class GetArtDataset(Dataset):
    def __init__(self, basedir, transforms, train=True):
        self.imgdir = os.path.join(basedir, "images")
        self.labdir = os.path.join(basedir, "labels")
        self.transforms = transforms
        self.train = train #if False: test
        self.labels = []
        self.images= []
        if self.train:
            file = os.path.join(self.labdir, "train.txt")
        else:
            file = os.path.join(self.labdir, "test.txt")
        f=open(file,"r")
        lines=f.readlines()
        for x in lines:
            self.images.append(x.split(' ')[0])
            self.labels.append(x.split(' ')[1])
        f.close()

    def __getitem__(self,idx):
        name = self.images[idx]
        image = os.path.join(self.imgdir, name)
        image = Image.open(image)
        image = self.transforms(image)
        label = int(self.labels[idx])
        return image, label

    def __len__(self):
        return len(self.images)


