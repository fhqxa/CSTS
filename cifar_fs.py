import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CifarFS(Dataset):

    def __init__(self, partition):
        TRAIN_PATH = './data/cifar_fs/meta-train'
        VAL_PATH = './data/cifar_fs/meta-val'
        TEST_PATH = './data/cifar_fs/meta-test'
        if partition == 'train':
            THE_PATH = TRAIN_PATH
        elif partition == 'test':
            THE_PATH = TEST_PATH
        elif partition == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong partition.')
        
        coarse_folders = [osp.join(THE_PATH, coarse_label) for coarse_label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, coarse_label))]   # coarse class path
        
        fine_folders = [os.path.join(coarse_label,label) \
                for coarse_label in coarse_folders \
                if os.path.isdir(coarse_label) \
                for label in os.listdir(coarse_label)
                ]
        coarse_labels = np.array(range(len(coarse_folders)))
        coarse_labels = dict(zip(coarse_folders, coarse_labels))
        # 
        labels = np.array(range(len(fine_folders)))
        labels = dict(zip(fine_folders, labels))
        
        if partition == 'val' or partition == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif partition == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

        data, coarse_label, fine_label = [],[],[]
        for c in fine_folders:
            for x in os.listdir(c):
                path = os.path.join(c, x)
                data.append(self.transform(Image.open(path).convert('RGB')).numpy())
                coarse_label.append(coarse_labels['E:\\'+self.get_coarse_class(path)])
                fine_label.append(labels['E:\\'+self.get_class(path)])
            
        #粗类标签
        # coarse_label = [coarse_labels['E:\\'+self.get_coarse_class(x)] for x in data]
    
        # # 细类标签
        # fine_label = [labels['E:\\'+self.get_class(x)] for x in data]

        self.fine_class=labels
        self.data = torch.Tensor(data)
        self.coarse_label = np.array(coarse_label)
        self.label = fine_label
        self.num_class = len(set(fine_label))
        self.num_coarse_class = len(set(coarse_label))

        # Transformation
        

    def get_class(self, sample):
        return os.path.join(*sample.split('\\')[1:-1])

    def get_coarse_class(self, sample):
        # print(*sample.split('\\'))
        return os.path.join(*sample.split('\\')[1:-2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, coarse_label,label = self.data[i], self.coarse_label[i],self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label,coarse_label


if __name__ == '__main__':
    pass