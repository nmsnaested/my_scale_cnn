#pylint: disable=E1101
import os, sys
import os.path
import gzip
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


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

