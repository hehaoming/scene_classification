import torch
import torch.nn.functional
import main.deep_learning.config as config
from glob import glob
import os
import cv2
import numpy as np
import random
from main.deep_learning.data_preprocessing.utils import NameMapID
from torch.utils.data import Dataset, DataLoader


# class TrainDataLoader(Sequence):
#     def __init__(self, batch_size):
#         self.name_map = NameMapID()
#         self.file_name = glob(os.path.join(config.DATA_PATH, "train", "*/*.jpg"))
#         random.shuffle(self.file_name)
#         self.batch_size = batch_size
#
#     def __getitem__(self, index):
#         origin_image = np.empty((self.batch_size, 256, 256, 3))
#         edge_image = np.empty((self.batch_size, 256, 256, 3))
#         label = np.empty((self.batch_size, 45))
#         for ind, file in enumerate(self.file_name[index * self.batch_size: (index + 1) * self.batch_size]):
#             origin_image[ind] = cv2.resize(cv2.imread(file), (256, 256))
#             name = file.lstrip(os.path.join(config.DATA_PATH, "train"))
#             edge_image[ind] = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "train", name))
#             label[ind] = to_categorical(self.name_map.name_to_label(name), num_classes=45)
#         # return [origin_image, edge_image], label
#
#     def __len__(self):
#         return int(np.ceil(len(self.file_name) / self.batch_size))
#
#     def __iter__(self):
#         return super().__iter__()
#
#
# class TestDataLoader(Sequence):
#     def __init__(self, batch_size):
#         self.name_map = NameMapID()
#         self.file_name = glob(os.path.join(config.DATA_PATH, "test", "*/*.jpg"))
#         random.shuffle(self.file_name)
#         self.batch_size = batch_size
#
#     def __getitem__(self, index):
#         origin_image = np.empty((self.batch_size, 256, 256, 3))
#         edge_image = np.empty((self.batch_size, 256, 256, 3))
#         label = np.empty((self.batch_size, 45))
#         for ind, file in enumerate(self.file_name[index * self.batch_size: (index + 1) * self.batch_size]):
#             origin_image[ind] = cv2.resize(cv2.imread(file), (256, 256))
#             name = file.lstrip(os.path.join(config.DATA_PATH, "test"))
#             edge_image[ind] = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "test", name))
#             label[ind] = to_categorical(self.name_map.name_to_label(name), num_classes=45)
#         return [origin_image, edge_image], label
#
#     def __len__(self):
#         return int(np.ceil(len(self.file_name) / self.batch_size))
#
#     def __iter__(self):
#         return super().__iter__()


class TrainDataLoaderPT(Dataset):
    def __init__(self):
        self.name_map = NameMapID()
        self.file_name = glob(os.path.join(config.DATA_PATH, "train", "*/*.jpg"))
        random.shuffle(self.file_name)

    def __getitem__(self, index):
        # origin_image = cv2.resize(cv2.imread(self.file_name[index]), (256, 256))
        # origin_image = np.transpose(origin_image, [2, 0, 1])
        # name = self.file_name[index].lstrip(os.path.join(config.DATA_PATH, "train"))
        # edge_image = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "train", name))
        # label = self.name_map.name_to_label(name)
        # return origin_image, label
        origin_image = cv2.resize(cv2.imread(self.file_name[index]), (256, 256))
        name = self.file_name[index].lstrip(os.path.join(config.DATA_PATH, "train"))
        edge_image = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "train", name))
        label = self.name_map.name_to_label(name)
        image = np.concatenate((origin_image, edge_image), axis=-1)
        image = np.transpose(image, [2, 0, 1])
        return image, label

    def __len__(self):
        return len(self.file_name)


class TestDataLoaderPT(Dataset):
    def __init__(self):
        self.name_map = NameMapID()
        self.file_name = glob(os.path.join(config.DATA_PATH, "test", "*/*.jpg"))
        random.shuffle(self.file_name)

    def __getitem__(self, index):
        origin_image = cv2.resize(cv2.imread(self.file_name[index]), (256, 256))
        origin_image = np.transpose(origin_image, [2, 0, 1])
        name = self.file_name[index].lstrip(os.path.join(config.DATA_PATH, "test"))
        # edge_image = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "test", name))
        label = self.name_map.name_to_label(name)
        return origin_image, label

    def __len__(self):
        return len(self.file_name)
