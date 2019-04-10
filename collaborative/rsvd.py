import numpy as np
import feature_prepare as fp
import random
import math
import error_handle as eh


def train_model(matrix, p, q, penalty, error=0.1, iter_times=10000, step=0.005):
    """
   fill the sparse matrix by svd
   :param matrix: sparse matrix
   :param p:
   :param u:
   :param penalty: penalty factory
   :param error: Convergence error
   :param iter_times: max iter times
   :return:
   """
    m, n = matrix.shape
    kl = p.shape[1]

    for times in range(iter_times):
        end_loop_flag = True  # if error of every data is smaller than Convergence error, end the loop

        for row in range(m):
            for column in range(n):
                if matrix[row, column] == 0:
                    continue

                e = matrix[row, column] - np.dot(p[row, :], q[:, column])  # error between actual value and prediction

                # if error is smaller than Convergence error
                if abs(e) <= float(abs(e - matrix[row, column])) / float(matrix[row, column]):
                    continue

                end_loop_flag = False

                # update the data in p and u
                p[row, :] += step * (e * q[:, column] - penalty * p[row, :])
                q[:, column] += step * (e * p[row, :] - penalty * q[:, column])
                # for k in range(kl):
                #
                #     p[row,k]+=step*(e*q[k,column]-penalty*p[row,k])
                #     q[k,column]+=step*(e*p[row,k]-penalty*q[column,k])

        if end_loop_flag:
            break

        step *=0.000008

    return np.dot(p, q)


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
                f.write(str(matirx[i, j]))
            else:
                f.write('\t' + str(matirx[i, j]))

        f.write('\n')
    f.close()


if __name__ == '__main__':
    train_matrix = fp.get_data2_matrix("../resources/data/clean_data_2/")
    train_matrix = train_matrix[0:200, :]
    m, n = train_matrix.shape
    k = int((m + n) / 2)

    f=math.sqrt(k)
    p = np.ones((m, k))
    for i in range(m):
        p[i] = np.array([0.1 * random.random() / f])

    q = np.ones((k, n))
    for i in range(k):
        q[i] = np.array([0.1 * random.random() / f])
    # p=np.full((m,k),1.5)
    # q=np.full((k,n),1.5)

    dense_matrix = train_model(train_matrix, p, q, 0.6)

    predict = []
    label = []
    for i in range(m):
        for j in range(n):
            if train_matrix[i, j] != 0:
                label.append(train_matrix[i, j])
                predict.append(dense_matrix[i, j])

    result, rate = eh.error_rate(predict, label, 0.1)

    print(rate)
    # save_matrix("../resources/data/svd/", "test", dense_matrix)
