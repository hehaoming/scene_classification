from keras.utils import Sequence, to_categorical
import main.deep_learning.config as config
from glob import glob
import os
import cv2
import numpy as np
import random
from main.deep_learning.data_preprocessing.utils import NameMapID


class TrainDataLoader(Sequence):
    def __init__(self, batch_size):
        self.name_map = NameMapID()
        self.file_name = glob(os.path.join(config.DATA_PATH, "train", "*/*.jpg"))
        random.shuffle(self.file_name)
        self.batch_size = batch_size

    def __getitem__(self, index):
        origin_image = np.empty((self.batch_size, 256, 256, 3))
        edge_image = np.empty((self.batch_size, 256, 256, 3))
        label = np.empty((self.batch_size, 45))
        for ind, file in enumerate(self.file_name[index * self.batch_size: (index + 1) * self.batch_size]):
            origin_image[ind] = cv2.resize(cv2.imread(file), (256, 256))
            name = file.lstrip(os.path.join(config.DATA_PATH, "train"))
            edge_image[ind] = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "train", name))
            label[ind] = to_categorical(self.name_map.name_to_label(name), num_classes=45)
        return [origin_image, edge_image], label

    def __len__(self):
        return int(np.ceil(len(self.file_name) / self.batch_size))

    def __iter__(self):
        return super().__iter__()


class TestDataLoader(Sequence):
    def __init__(self, batch_size):
        self.name_map = NameMapID()
        self.file_name = glob(os.path.join(config.DATA_PATH, "test", "*/*.jpg"))
        random.shuffle(self.file_name)
        self.batch_size = batch_size

    def __getitem__(self, index):
        origin_image = np.empty((self.batch_size, 256, 256, 3))
        edge_image = np.empty((self.batch_size, 256, 256, 3))
        label = np.empty((self.batch_size, 45))
        for ind, file in enumerate(self.file_name[index * self.batch_size: (index + 1) * self.batch_size]):
            origin_image[ind] = cv2.resize(cv2.imread(file), (256, 256))
            name = file.lstrip(os.path.join(config.DATA_PATH, "test"))
            edge_image[ind] = cv2.imread(os.path.join(config.EDGE_DATA_PATH, "test", name))
            label[ind] = to_categorical(self.name_map.name_to_label(name), num_classes=45)
        return [origin_image, edge_image], label

    def __len__(self):
        return int(np.ceil(len(self.file_name) / self.batch_size))

    def __iter__(self):
        return super().__iter__()
