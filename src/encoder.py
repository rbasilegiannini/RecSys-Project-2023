import numpy as np


def get_binary_encoding(dim, index_elements):
    """
    :param dim:
        The array's length
    :param index_elements:
        Indexes to be set to 1
    :return:
        The binary vector
    """
    binary_vector = np.zeros(dim, int)
    for idx in index_elements:
        binary_vector[idx] = 1

    return binary_vector
