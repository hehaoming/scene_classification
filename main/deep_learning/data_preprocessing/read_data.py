import cv2
import os
import glob
import main.deep_learning.config as config
from main.deep_learning.data_preprocessing.utils import NameMapID


class DataSet:

    def __init__(self, is_train=True):
        self.name_map = NameMapID()
        if is_train:
            mode = config.TRAIN_PATH
        else:
            mode = config.TEST_PATH
        self.file_name = glob.glob(os.path.join(config.DATA_PATH, mode, "*/*.jpg"))

    def data_generator(self):
        """
        note: all label had changed to label - 1
        :return: a sample image and label
        """
        # print(file_name)
        for file in self.file_name:
            image = cv2.imread(file)
            # print(file)
            name = file.lstrip(os.path.join(config.DATA_PATH, mode))
            # print(name)
            label = self.name_map.name_to_label(name)
            # print(label)
            yield image, label

    def __len__(self):
        return len(self.file_name)


if __name__ == "__main__":
    data_set = DataSet(False)
    print(len(data_set))
    # for a_image, a_label in data_set.data_generator():
    #     print(a_image.shape, a_label)
