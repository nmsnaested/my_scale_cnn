#pylint: disable=E1101
import os, sys
import os.path
import gzip
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import PIL
from PIL import Image
from PIL import ImageMath

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

class ImgNetDataset(Dataset):
    def __init__(self, rootdir, mode, transforms=None):
        '''
        :basedir: base directory in which are all the dataset folders
        :mode: train, val (validation) or test
        :transforms: list of transformations to apply to the images
        '''
        self.root = rootdir
        self.dir = os.path.join(rootdir, mode)
        self.imgdir = os.path.join(self.dir, "images")
        self.mode = mode
        if mode != "train" and mode != "val" and mode != "test":
            raise ValueError("mode is {}, should be 'train', 'val' or 'test'".format(self.mode))

        self.images = [filename for filename in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, filename)) and filename!='.DS_Store']
        self.names = []
        filename = os.path.join(self.root, "wnids.txt")
        with open(filename, "rt") as f:
            for line in f.readlines():
                self.names.append(line.split('\n')[0])
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img = os.path.join(self.imgdir, self.images[idx])
        img = Image.open(img)
        if img.mode != 'RGB':
            if img.mode == 'I':
                img = ImageMath.eval('im/256', {'im': img})
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        if self.mode == "train": 
            name = self.images[idx].split('_')[0]
            label = self.names.index(name)

        elif self.mode == "val": 
            filename = os.path.join(self.dir, "val_annotations.txt")
            with open(filename, "rt") as f:
                line = f.readlines()[idx]
                name = line.split("\t")[1]
                label = next((idx for idx, n in self.names if name == n), None)

        elif self.mode == "test":
            label = None
        
        return img, label
        
    def __len__(self):
        return len(self.images)
