import numpy as np
from scipy.linalg import svd
import feature_prepare as fp
import error_handle as eh


def save_matrix(file_path, name, matirx):
    """
    save the matrix into .txt
    :param file_path:
    :param name:
    :param matirx:
    :return:
    """
    f = open(file_path + name + ".txt", 'w')
    m, n = matirx.shape

    for i in range(m):
        for j in range(n):
            if 0 == j:
                f.write(str(abs(matirx[i, j])))
            else:
                f.write('\t' + str(abs(matirx[i, j])))

        f.write('\n')
    f.close()


def cal_svd_matrix(matrix):
    """
    calculate the middle matrix of svd
    :param matrix:
    :return:
    """
    U, S, V = svd(D, full_matrices=True)
    V = V.T
    return U, S, V


def tailor_matrix(U, S, V, k):
    """
    to tailor the matrix
    :param U:
    :param S:
    :param V:
    :param k:
    :return:
    """
    if (k > len(U[0]) or k > len(S) or k > len(V[0])):
        print(" 你丫有病 ")

    U = U[:, :k]
    S = fill_tailor_s(S, k)
    V = V[:, :k]

    return U, S, V


def fill_tailor_s(S, k):
    """
    to fill the matrix,then failor it
    :param S:
    :param k:
    :return:
    """
    S = S[:k]
    NS = np.zeros((k, k))
    i = 0
    for item in S:
        NS[i][i] = item
        i += 1

    return NS


def comb_matrix(U, S, V):
    """
    to combine the matrix
    :param U:
    :param S:
    :param V:
    :return:
    """
    a = np.matmul(U, S)
    b = np.matmul(a, V.T)

    return b


def nor_tailor_matrix(matrix):
    """
    without the tailor
    :param matrix:
    :return:
    """
    U, S, V = svd(D, full_matrices=True)
    a = len(U[0])
    b = len(S)
    NS = np.zeros((a, b))
    i = 0
    for item in S:
        NS[i][i] = item
        i += 1
    a = np.matmul(U, NS)
    b = np.matmul(a, V)
    return b


if __name__ == '__main__':
    # 裁剪过
    # D = np.array([[5, 5, 0, 5], [5, 0, 3, 4], [3, 4, 0, 3], [0, 0, 5, 3], [5, 4, 4, 5], [5, 4, 5, 5]])
    D = fp.get_data2_matrix("../resources/data/test_data_2/")
    # D = D[0:200, :]

    m, n = D.shape
    U, S, V = cal_svd_matrix(D)
    U, S, V = tailor_matrix(U, S, V, 25)
    NB = comb_matrix(U, S, V)

    save_matrix("../resources/data/svd/","nb-test",NB)
    # 绝对值
    # print(np.abs(NB))
    predict = []
    label = []
    for i in range(m):
        for j in range(n):
            if D[i, j] != 0:
                label.append(D[i, j])
                predict.append(NB[i, j])

    result, rate = eh.error_rate(predict, label, 0.1)
    print(rate)
    # 未裁剪过
    B = nor_tailor_matrix(D)
    save_matrix("../resources/data/svd/", "b-test", B)
    predict = []
    label = []
    for i in range(m):
        for j in range(n):
            if D[i, j] != 0:
                label.append(D[i, j])
                predict.append(B[i, j])

    result, rate = eh.error_rate(predict, label, 0.1)
    print(rate)
    # print(B)
