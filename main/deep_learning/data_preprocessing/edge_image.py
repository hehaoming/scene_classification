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
        self.BASIC_PAT = config.EDGE_DATA_PATH

    @staticmethod
    def get_edge_image(image):
        edge_0 = cv2.Canny(image, 100, 200).reshape(256, 256, 1)

        edge_1 = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5).reshape(256, 256, 1)
        edge_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel64f = np.absolute(edge_sobel)
        edge_2 = np.uint8(abs_sobel64f).reshape(256, 256, 1)
        # plt.subplot(121)
        # plt.imshow(image, cmap='gray')
        # plt.title('Original Image')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(122)
        # plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        edges = np.concatenate((edge_0, edge_1, edge_2), axis=-1)
        return edges

    def image_to_edge(self, is_train=True):
        dataset = DataSet(is_train)
        for (img, label, name) in tqdm(dataset.data_generator_gray(), total=len(dataset)):
            edge_image = self.get_edge_image(img)
            class_name = self.name_map.label_to_name(label)
            if not os.path.exists(os.path.join(self.BASIC_PAT, dataset.mode, class_name)):
                os.makedirs(os.path.join(self.BASIC_PAT, dataset.mode, class_name))
            cv2.imwrite(os.path.join(self.BASIC_PAT, dataset.mode, name), edge_image)


if __name__ == "__main__":
    image_gen = EdgeImageGenerator()
    image_gen.image_to_edge(True)
