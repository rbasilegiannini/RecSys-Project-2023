import numpy as np


def get_matrix_from_list(lst, element_size):
    matrix = np.zeros((len(lst), element_size))
    for i in range(len(lst)):
        matrix[i, :] = lst[i]

    return matrix


def remove_duplicates(lst):
    no_duplicates_list = []
    [no_duplicates_list.append(element) for element in lst if element not in lst]

    return no_duplicates_list
