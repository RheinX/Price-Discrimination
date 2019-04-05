import feature_prepare as fp
import numpy as np

def matrix_reduce(vector, matrix):
    """
    delete the column which value in of vecotor is 0 of matrix
    :param vector:
    :param matrix:
    :return:
    """
    delete_column=np.where(vector!=0)
