import numpy as np
import os


class DatasetExtractor:
    def __init__(self, users_dimension, items_dimension):
        self.users_size = users_dimension
        self.items_size = items_dimension
        self.test_items = []

    def get_urm(self):
        dataset_entries = self.get_dataset_entries()
        self.choose_test_items_from_dataset(dataset_entries)
        urm = self.fill_urm(dataset_entries)
        urm = self.reset_test_items(urm)
        return urm

    def get_dataset_entries(self):
        absolute_path = os.path.dirname(__file__)
        dataset_path = absolute_path.replace("src", "res")
        dataset_path = os.path.join(dataset_path, "u.data")

        dataset_file = open(dataset_path, 'r')
        dataset_text = dataset_file.read()
        return dataset_text.split("\n")

    def choose_test_items_from_dataset(self, dataset_entries):
        interaction_timestamps = np.zeros((self.users_size, self.items_size))

        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '' and entry_fields[3] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                timestamp = int(entry_fields[3])

                interaction_timestamps[user_id - 1][item_id - 1] = timestamp

        self.test_items = np.argmax(interaction_timestamps, axis=1)

    def fill_urm(self, dataset_entries):
        urm = np.zeros((self.users_size, self.items_size), dtype=int)
        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                urm[user_id - 1][item_id - 1] = 1
        return urm

    def reset_test_items(self, urm):
        for user in range(self.users_size):
            urm[user, self.test_items[user]] = 0
        return urm

    def get_test_items(self):
        return self.test_items
