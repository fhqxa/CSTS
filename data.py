from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
import pickle
from itertools import islice
from torchvision import transforms
from   tqdm import tqdm
import cv2

class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, 'miniImageNet', 'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        # for each class
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                #image_data = np.array(image_data, dtype='float32')

                #image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=5,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, support_coarse_label, query_data, query_label, query_coarse_label = [], [], [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
            support_coarse_label.append(label.copy())
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)
            query_coarse_label.append(label.copy())

        # get full class list in dataset
        full_class_list = list(self.data.keys())
        #10
        full_coarse_class_list = np.array([7, 2, 0, 0, 0, 0, 3, 0, 3, 0, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 3, 0,
                                 3, 6, 2, 6, 7, 1, 8, 6, 2, 7, 7, 8, 5, 1, 8, 2, 1, 1, 6, 2, 8, 2, 1,
                                1, 1, 1, 7, 6, 8, 2, 6, 1, 5, 2, 2, 7, 7, 3, 7, 2, 3, 3, 12, 12, 11,
                                 11, 11, 11, 11, 12, 11, 10, 14, 10, 10, 10, 14, 10, 10, 10, 14, 13,
                                  17, 19, 19, 19, 17, 15, 16, 15, 15, 15, 15, 15, 18, 15, 18, 16])
        #12
        # full_coarse_class_list = np.array([1, 2, 10, 10, 3, 3, 0, 10, 0, 3, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 0, 3,
        #                                    0, 6, 2, 6, 1, 7, 8, 6, 2, 1, 1, 8, 11, 5, 8, 2, 7, 7, 6, 2, 8, 2, 5,
        #                                    5, 5, 5, 1, 6, 8, 2, 6, 7, 11, 2, 2, 1, 1, 0, 1, 2, 0, 0, 14, 14, 13, 13,
        #                                    13, 13, 13, 14, 13, 12, 16, 12, 12, 12, 16, 12, 12, 12, 16, 15, 19, 21, 21,
        #                                    21, 19, 17, 18, 17, 17, 17, 17, 17, 20, 17, 20, 18])
        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)
            class_coarse_list = full_coarse_class_list[task_class_list]
            # print(task_class_list)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = task_class_list[c_idx]
                    support_coarse_label[i_idx + c_idx * num_shots][t_idx] = class_coarse_list[c_idx]

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = task_class_list[c_idx]
                    query_coarse_label[i_idx + c_idx * num_queries][t_idx] = class_coarse_list[c_idx]

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        support_coarse_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_coarse_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)
        query_coarse_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_coarse_label], 1)

        # return [support_data, support_label, query_data, query_label]

        return [support_data, support_coarse_label, support_label, query_data, query_coarse_label, query_label]

class TieredImagenetLoader(object):
    def __init__(self, root, partition='train'):
        super(TieredImagenetLoader, self).__init__()
        self.root = root
        self.partition = partition  # train/val/test
        self.data_size = [3, 84, 84]

        # load data
        self.data, self.data_coarse_labels = self.load_data_pickle()

    def load_data_pickle(self):

        print("Loading dataset")
        labels_name = '{}/tiered-imagenet/{}_labels.pkl'.format(self.root, self.partition)
        images_name = '{}/tiered-imagenet/{}_images.zip'.format(self.root, self.partition)
        print('labels:', labels_name)
        print('images:', images_name)

        # decompress images if npz not exits
        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            # print(png_pkl)
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')

        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name) as f:
                    data = pickle.load(f)
                    label_specific = data["label_specific"]
                    label_general = data["label_general"]
            except:
                with open(labels_name, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    label_specific = data['label_specific']
                    label_general = data["label_general"]
            print('read label data:{}'.format(len(label_specific)))
        labels = label_specific
        coarse_labels = label_general
        # print(images_name)
        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]
            print('read image data:{}'.format(image_data.shape))

        data = {}
        data_coarse_labels = []
        n_classes = np.max(labels) + 1
        for c_idx in range(n_classes):
            data[c_idx] = []
            idxs = np.where(labels==c_idx)[0]
            data_coarse_labels.append(coarse_labels[idxs[0]])
            np.random.RandomState(tt.arg.seed).shuffle(idxs)  # fix the seed to keep label,unlabel fixed
            for i in idxs:
                image2resize = pil_image.fromarray(np.uint8(image_data[i,:,:,:]))
                image_resized = image2resize.resize((self.data_size[2], self.data_size[1]))
                image_resized = np.array(image_resized, dtype='float32')

                # Normalize
                image_resized = np.transpose(image_resized, (2, 0, 1))
                image_resized[0, :, :] -= 120.45  # R
                image_resized[1, :, :] -= 115.74  # G
                image_resized[2, :, :] -= 104.65  # B
                image_resized /= 127.5
                data[c_idx].append(image_resized)
        return data, data_coarse_labels

    def get_task_batch(self,
                       num_tasks,
                       num_ways,
                       num_shots,
                       num_queries,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_coarse_label, support_label, query_data, query_coarse_label, query_label = [], [], [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_coarse_label.append(label)
            support_label.append(label.copy())

        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_coarse_label.append(label)
            query_label.append(label.copy())

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)
            task_class_coarse_labels = np.array(self.data_coarse_labels)[task_class_list]
            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = task_class_list[c_idx]
                    support_coarse_label[i_idx + c_idx * num_shots][t_idx] = task_class_coarse_labels[c_idx]


                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_coarse_label[i_idx + c_idx * num_queries][t_idx] = task_class_coarse_labels[c_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = task_class_list[c_idx]

        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_coarse_label = torch.stack([torch.from_numpy(coarse_label).float().to(tt.arg.device) for coarse_label in support_coarse_label], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_coarse_label = torch.stack([torch.from_numpy(coarse_label).float().to(tt.arg.device) for coarse_label in query_coarse_label], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)
        if self.partition == 'val':
            support_coarse_label += 20
            query_coarse_label += 20
        if self.partition == 'test':
            support_coarse_label += 26
            query_coarse_label += 26
            
        return [support_data, support_coarse_label, support_label, query_data, query_coarse_label, query_label]

def compress(path, output):
  with np.load(path, mmap_mode="r") as data:
    images = data["images"]
    array = []
    for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
      im = images[ii]
      im_str = cv2.imencode('.png', im)[1]
      array.append(im_str)
  with open(output, 'wb') as f:
    pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)


def decompress(path, output):
  with open(output, 'rb') as f:
    array = pickle.load(f, encoding='bytes')
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)