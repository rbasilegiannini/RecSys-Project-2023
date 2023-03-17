import numpy as np


def get_onehot_encoding(size):
    onehot_matrix = np.identity(size, dtype=int)
    return onehot_matrix
    
    
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


def get_neighborhoods_encoding(neighborhoods, size):
    """
    :param neighborhoods:
        The neighborhoods
    :param size:
        The array's length
    :return:
        The neighborhoods in binary encoding
    """
    binary_neighborhoods = []
    for neighborhood in neighborhoods:
        binary_neighborhoods.append(
            get_binary_encoding(size, neighborhood)
        )

    return binary_neighborhoods

