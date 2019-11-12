import cv2
import os
import numpy as np
import main.deep_learning.config as config
from tqdm import tqdm
from matplotlib import pyplot as plt
from main.deep_learning.data_preprocessing.utils import NameMapID
from main.deep_learning.data_preprocessing.read_data import DataSet


class EdgeImageGenerator:
    def __init__(self):
        self.name_map = NameMapID()
        self.BASIC_PATH = config.COLOR_DATA_PATH

    @staticmethod
    def get_edge_image(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # hist.shape = (180, 256)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hist

    def image_to_color(self, is_train=True):
        dataset = DataSet(is_train)
        for (img, label, name) in tqdm(dataset.data_generator(), total=len(dataset)):
            edge_image = self.get_edge_image(img)
            class_name = self.name_map.label_to_name(label)
            if not os.path.exists(os.path.join(self.BASIC_PATH, dataset.mode, class_name)):
                os.makedirs(os.path.join(self.BASIC_PATH, dataset.mode, class_name))
            cv2.imwrite(os.path.join(self.BASIC_PATH, dataset.mode, name), edge_image)


if __name__ == "__main__":
    image_gen = EdgeImageGenerator()
    image_gen.image_to_color()
