import numpy as np
import random


def get_nearest_indices(vector, matrix, num=5):
    """
    return the indice of random nearest neighbors
    distance is Manhattan distance
    :param vector:
    :param matrix:
    :return:
    """
    if len(matrix) < num:
        num = len(matrix)

    m, n = matrix.shape
    diff_matrix = np.tile(vector, (m, 1)) - matrix
    diff_matrix = abs(diff_matrix)

    distance = diff_matrix.sum(axis=1)
    sortIndices = np.argsort(distance)
    sortIndices = sortIndices[0:num]

    return sortIndices[random.randint(0, num - 1)]


def generate_data(v1, v2, l1, l2):
    """
    generate a new sample by smote
    :param v1:
    :param v2:
    :return:
    """
    alpha = random.random()
    vector = v1 + alpha * abs(v1 - v2)
    label = l1 + alpha * abs(l1 - l2)

    return vector, label


def fill_matrix(matrix, label, num):
    """
    generate new data until length is equal num
    :param matrix:
    :param label:
    :param num:
    :return:
    """
    m, n = matrix.shape
    mat = np.zeros((num, n))
    label=list(label)
    for i in range(num):
        if i < m:
            mat[i, :] = matrix[i, :]

        # get a random vector
        else:
            vector_id = random.randint(0, i-1)
            vector1 = mat[vector_id, :]
            l1 = label[vector_id]

            # get a neighbors
            nid = get_nearest_indices(l1, mat[0:i, :])
            v2 = mat[nid, :]
            l2 = label[nid]

            n_vector, n_label = generate_data(vector1, v2, l1, l2)

            mat[i,:]=n_vector
            label.append(n_label)

    return mat,np.array(label)
