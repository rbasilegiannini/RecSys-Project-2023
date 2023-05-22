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

    def __compute_urm(self):
        dataset_entries = self.__read_dataset_from_file()
        self.__fill_urm(dataset_entries)
        self.__choose_test_items_from_dataset(dataset_entries)
        self.__reset_test_items_in_urm()

    def __read_dataset_from_file(self):
        absolute_path = os.path.dirname(__file__)
        dataset_path = absolute_path.replace("src", "res")
        dataset_path = os.path.join(dataset_path, "u.data")
        dataset_file = open(dataset_path, 'r')
        dataset_text = dataset_file.read()
        dataset_entries = dataset_text.split("\n")

        return dataset_entries

    def __fill_urm(self, dataset_entries):
        urm = np.zeros((self.__users_size, self.__items_size), dtype=int)
        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                if user_id <= self.__users_size and item_id <= self.__items_size:
                    urm[user_id - 1][item_id - 1] = 1
        self.__urm = urm

    def __choose_test_items_from_dataset(self, dataset_entries):
        interaction_timestamps = np.zeros((self.__users_size, self.__items_size))

        for entry in dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '' and entry_fields[3] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                timestamp = int(entry_fields[3])

                if user_id <= self.__users_size and item_id <= self.__items_size:
                    interaction_timestamps[user_id - 1][item_id - 1] = timestamp

        self.__test_items = np.argmax(interaction_timestamps, axis=1)

    def __reset_test_items_in_urm(self):
        for user in range(self.__users_size):
            self.__urm[user, self.__test_items[user]] = 0

    def get_urm(self):
        return self.__urm

    def get_test_items(self):
        return self.__test_items

    def get_not_interacted_items(self, user):
        items = np.where(self.__urm[user] == 0)[0]
        return items

    def get_not_interacted_items_for_recommendation(self, user):
        user_not_interacted_items = self.get_not_interacted_items(user)
        user_test_item = self.__test_items[user]
        user_not_interacted_items = user_not_interacted_items[user_not_interacted_items != user_test_item]

        selected_not_interacted_items = np.zeros(100 + 1, dtype=int)
        selected_not_interacted_items[:100] = np.random.choice(user_not_interacted_items, 100, replace=False)
        selected_not_interacted_items[100] = user_test_item

        return selected_not_interacted_items


