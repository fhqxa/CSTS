from __future__ import print_function
from torchtools import *
import os
import os.path as osp
import torch

import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUB(Dataset):

    def __init__(self, partition):
        IMAGE_PATH = os.path.join('./data/CUB/', 'cub/')
        SPLIT_PATH = os.path.join('./data/CUB/', 'cub/split/')
        txt_path = osp.join(SPLIT_PATH, partition + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        if partition == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        if partition == 'train':
            lines.pop(5864)  #this image file is broken

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(self.transform(Image.open(path).convert('RGB')).numpy())
            label.append(lb)

        self.data = torch.Tensor(data)
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        
        return image, label



            

if __name__ == '__main__':
    pass
