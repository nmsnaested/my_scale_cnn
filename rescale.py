import torch
import torchvision
from torchvision import transforms
from torchvision import datasets

import numpy as np

import random
from PIL import Image
import torchvision.transforms.functional as F


class RandomResizedCrop(object):
  
    def __init__(self, size, scale=(0.08, 1.0), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.scale = scale
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(*scale) * area

        w = int(round(target_area ** 0.5))
        w = min(img.size[0], img.size[1], w)
        i = (img.size[1] - w) // 2 #take crop at the center of the img 
        j = (img.size[0] - w) // 2
        return i, j, w, w #square img

    def __call__(self, img): 
        i, j, h, w = self.get_params(img, self.scale)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomRescale(object):
    
    def __init__(self, size, scales, sampling="uniform", interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.sampling = sampling
        self.scales = scales

    def __call__(self, img):
        s = -1
        while s < 0:
            if self.sampling == "uniform":
                s = np.random.uniform(*self.scales)
            elif self.sampling == "normal":
                s = np.random.normal(*self.scales)
        pad_h = int(round((img.size[0]/s - img.size[0])/2))
        pad_w = int(round((img.size[1]/s - img.size[1])/2))
        padding = transforms.Pad((pad_h, pad_w))
        img = padding(img)
        resize = transforms.Resize(self.size, self.interpolation)
        img = resize(img)
        
        return img