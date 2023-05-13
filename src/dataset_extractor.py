import numpy as np
import os


class DatasetExtractor:
    """
    This class is used to extract data from the dataset MovieLens.
    """

    def __init__(self, users_dimension, items_dimension):
        self.__users_size = users_dimension
        self.__items_size = items_dimension

        self.__urm = None
        self.__test_items = None

        self.__compute_urm()

    def __choose_test_items_from_dataset(self, dataset_entries):
        interaction_timestamps = np.zeros((self.__users_size, self.__items_size))

        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '' and entry_fields[3] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                timestamp = int(entry_fields[3])

                interaction_timestamps[user_id - 1][item_id - 1] = timestamp

        self.__test_items = np.argmax(interaction_timestamps, axis=1)

    def __fill_urm(self, dataset_entries):
        urm = np.zeros((self.__users_size, self.__items_size), dtype=int)
        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                urm[user_id - 1][item_id - 1] = 1
        self.__urm = urm

    def __reset_test_items(self):
        for user in range(self.__users_size):
            self.__urm[user, self.__test_items[user]] = 0

    def __compute_urm(self):
        absolute_path = os.path.dirname(__file__)
        dataset_path = absolute_path.replace("src", "res")
        dataset_path = os.path.join(dataset_path, "u.data")

        dataset_file = open(dataset_path, 'r')
        dataset_text = dataset_file.read()
        dataset_entries = dataset_text.split("\n")

        self.__choose_test_items_from_dataset(dataset_entries)
        self.__fill_urm(dataset_entries)
        self.__reset_test_items()

    # Interface

    def get_urm(self):
        return self.__urm

    def get_test_items(self):
        return self.__test_items

    def get_non_interacted_items(self, user):
        items = list(np.where(self.__urm[user] == 0)[0])
        return items
