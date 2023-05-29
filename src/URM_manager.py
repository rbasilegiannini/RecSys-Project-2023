import numpy as np
import dataset_extractor as ds_extractor


class URMManager:
    def __init__(self, users_size, items_size):
        """

        :param users_size:
            The number of users.
        :param items_size:
            The number of items.

        """
        self.__users_size = users_size
        self.__items_size = items_size
        self.__urm = None
        self.__test_items = None
        self.__dataset_extractor = ds_extractor.DatasetExtractor(self.__users_size, self.__items_size)

    def get_urm(self):
        if self.__urm is None:
            self.__urm = self.__dataset_extractor.extract_urm_from_dataset()
            self.__compute_test_items()
        return self.__urm

    def get_test_items(self):
        if self.__urm is None:
            raise RuntimeError("URM not instantiated")
        if self.__test_items is None:
            self.__compute_test_items()

        return self.__test_items

    def __compute_test_items(self):
        interaction_timestamps = self.__dataset_extractor.get_interaction_timestamps_from_dataset()
        self.__test_items = np.argmax(interaction_timestamps, axis=1)

        self.__reset_test_items_in_urm()

    def __reset_test_items_in_urm(self):
        for user in range(self.__users_size):
            self.__urm[user, self.__test_items[user]] = 0

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
