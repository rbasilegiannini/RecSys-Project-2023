import numpy as np
import os


class DatasetExtractor:
    """
    This class is used to extract data from the dataset MovieLens.
    """

    def __init__(self, users_size, items_size):
        """

        :param users_size:
            The number of users.
        :param items_size:
            The number of items.

        """
        self.__users_size = users_size
        self.__items_size = items_size
        self.__dataset_entries = None

    def extract_urm_from_dataset(self):
        """
        Public method for URM extraction request.

        :return:
            The extracted URM from dataset.
        """
        self.__dataset_entries = self.__read_dataset_from_file()
        urm = self.__fill_urm()

        return urm

    def __read_dataset_from_file(self):
        """
        Computes the dataset file path and reads the MovieLens dataset.

        :return:
            The dataset interactions entries, formatted as:
            <user id> <item id> <rating> <timestamp>
        """
        absolute_path = os.path.dirname(__file__)
        dataset_path = absolute_path.replace("src", "res")
        dataset_path = os.path.join(dataset_path, "u.data")
        dataset_file = open(dataset_path, 'r')
        dataset_text = dataset_file.read()
        dataset_entries = dataset_text.split("\n")

        return dataset_entries

    def __fill_urm(self):
        """
        Fills the URM using the read dataset entries:

        :return:
            The extracted URM from dataset.
            urm[user][item] = 1 when an interaction occured, urm[user][item] = 1 otherwise.
        """
        urm = np.zeros((self.__users_size, self.__items_size), dtype=int)
        for entry in self.__dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                urm[user_id - 1][item_id - 1] = 1
        return urm

    def get_interaction_timestamps_from_dataset(self):
        """
        Public method for interactions timestamps request.
        Fills the interaction timestamps matrix.

        :return:
            The computed interaction timestamp matrix.
            interaction_timestamps[user][item] = timestamp when user interacted with item.

        """
        interaction_timestamps = np.zeros((self.__users_size, self.__items_size),dtype=int)

        for entry in self.__dataset_entries:
            entry_fields = entry.split("\t")
            if entry_fields[0] != '' and entry_fields[1] != '' and entry_fields[3] != '':
                user_id = int(entry_fields[0])
                item_id = int(entry_fields[1])
                timestamp = int(entry_fields[3])

                interaction_timestamps[user_id - 1][item_id - 1] = timestamp

        return interaction_timestamps








