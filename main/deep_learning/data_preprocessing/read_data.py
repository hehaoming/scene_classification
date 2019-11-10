import cv2
import os
import glob
import numpy as np
import main.deep_learning.config as config
from main.deep_learning.data_preprocessing.utils import NameMapID


class DataSet:

    def __init__(self, is_train=True):
        self.name_map = NameMapID()
        if is_train:
            self.mode = config.TRAIN_PATH
        else:
            self.mode = config.TEST_PATH
        self.file_name = glob.glob(os.path.join(config.DATA_PATH, self.mode, "*/*.jpg"))

    def data_generator(self):
        """
        note: all label had changed to label - 1
        :return: a sample image and label
        """
        # print(file_name)
        for file in self.file_name:
            image = cv2.imread(file)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            # print(file)
            name = file.lstrip(os.path.join(config.DATA_PATH, self.mode))
            # print(name)
            label = self.name_map.name_to_label(name)
            # print(label)
            yield image, label

    def data_generator_gray(self):
        """
        note: all label had changed to label - 1
        :return: a sample image that convert to gray image and label
        """
        # print(file_name)
        for file in self.file_name:
            image = cv2.imread(file, 0)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            # print(file)
            name = file.lstrip(os.path.join(config.DATA_PATH, self.mode))
            # print(name)
            label = self.name_map.name_to_label(name)
            # print(label)
            yield image, label, name

    def count_num_each_class(self):
        classes = self.name_map.classes()
        # count = [0] * len(classes)
        # count_map = zip(classes, count)
        # print(dict(count_map))
        for cla in classes:
            cla_list = glob.glob(os.path.join(config.DATA_PATH, self.mode, cla, "*.jpg"))
            print(cla, ": ", len(cla_list))

    def data_generator_with_edge(self):
        """
        note: all label had changed to label - 1
        :return: a sample image with edge image and label
        """
        # print(file_name)
        for file in self.file_name:
            origin_image = cv2.imread(file)
            # print(file)
            name = file.lstrip(os.path.join(config.DATA_PATH, self.mode))
            # print(name)
            edge_image = cv2.imread(os.path.join(config.EDGE_DATA_PATH, self.mode, name))
            # image = np.concatenate((origin_image, edge_image), axis=-1)
            label = self.name_map.name_to_label(name)
            # print(label)
            yield [origin_image, edge_image], label

    def __len__(self):
        return len(self.file_name)


if __name__ == "__main__":
    data_set = DataSet(True)
    # data_set.count_num_each_class()
    for a_image, a_label in data_set.data_generator_with_edge():
        print(a_image.shape, a_label)
