import numpy as np
import os


class DatasetExtractor:
    def __init__(self, users_dimension, items_dimension):
        self.users_size = users_dimension
        self.items_size = items_dimension

    def get_urm(self):
        dataset_entries = self.get_dataset_entries()
        urm = self.fill_urm(dataset_entries)
        return urm

    def fill_urm(self, dataset_entries):
        urm = np.zeros((self.users_size, self.items_size), dtype=int)
        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                urm[user_id - 1][item_id - 1] = 1
        return urm

    def get_dataset_entries(self):
        absolute_path = os.path.dirname(__file__)
        dataset_path = absolute_path.replace("src", "res")
        dataset_path = os.path.join(dataset_path, "u.data")

        dataset_file = open(dataset_path, 'r')
        dataset_text = dataset_file.read()
        return dataset_text.split("\n")

