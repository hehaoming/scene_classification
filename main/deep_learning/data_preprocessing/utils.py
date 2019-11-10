import main.deep_learning.config as config
import os


class NameMapID:
    def __init__(self):
        self.name_id_map = {}
        self.id_name_map = {}
        with open(os.path.join(config.DATA_PATH, config.NAME_CONVERT_ID), 'r') as file:
            for line in file.readlines():
                name_list = line.split(':')
                self.name_id_map[name_list[0]] = name_list[2].rstrip('\n')
                self.id_name_map[name_list[2].rstrip('\n')] = name_list[0]

    def name_to_label(self, file_name):
        class_name, _ = file_name.split('/')
        return int(self.name_id_map[class_name]) - 1
        # print(class_name)

    def label_to_name(self, label):
        return self.id_name_map[str(int(label) + 1)]

    def classes(self):
        return list(self.name_id_map)


if __name__ == "__main__":
    namemap = NameMapID()
    print(namemap.id_name_map)
