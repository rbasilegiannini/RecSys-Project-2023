import numpy as np


def get_matrix_from_list(list, element_size):
    matrix = np.zeros((len(list), element_size))
    for i in range(len(list)):
        matrix[i, :] = list[i]

    return matrix
