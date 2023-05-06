import numpy as np


def get_matrix_from_list(list, element_size):
    matrix = np.zeros((len(list), element_size))
    for i in range(len(list)):
        matrix[i, :] = list[i]

    return matrix


def remove_duplicates(list):
    no_duplicates_list = []
    [no_duplicates_list.append(element) for element in list if element not in list]

    return no_duplicates_list
