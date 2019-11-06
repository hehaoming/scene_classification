import cv2
import os
import glob
import main.deep_learning.config as config
from main.deep_learning.data_preprocessing.utils import NameMapID


class DataSet:

    def __init__(self):
        self.name_map = NameMapID()

    def train_data_generator(self):
        """
        note: all label had changed to label - 1
        :return: a sample image and label
        """
        file_name = glob.glob(os.path.join(config.DATA_PATH, config.TRAIN_PATH, "*/*.jpg"))
        # print(file_name)
        for file in file_name:
            image = cv2.imread(file)
            # print(file)
            name = file.lstrip(os.path.join(config.DATA_PATH, config.TRAIN_PATH))
            # print(name)
            label = self.name_map.name_to_label(name)
            # print(label)

            yield image, label


if __name__ == "__main__":
    data_set = DataSet()
    for image, label in data_set.train_data_generator():
        print(image.shape, label)
