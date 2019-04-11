import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import feature_prepare as fp


def item_cluster_show():
    """
    show the cluster of item by their avg price
    :return:
    """
    items = fp.get_item_price("resources/data/clean_data_2/item.txt")

    plt_x = []
    plt_y = []

    for iid in items:
        plt_x.append(iid)
        plt_y.append(items[iid])

    plt.scatter(plt_x, plt_y)
    plt.show()


def user_cluster_show():
    """
    use user_item matrix to show the situation of cluster
    :return:
    """
    matrix = fp.get_data2_matrix("../resources/data/clean_data_2/")
    plt_x = []
    ply_y = []

    m, n = matrix.shape

    for uid in range(m):
        for iid in range(n):
            if matrix[uid, iid] != 0:
                plt_x.append(uid)
                ply_y.append(matrix[uid, iid])

    plt.scatter(plt_x, ply_y)
    plt.show()


def meanshift_cluser(file_path, bandwidth):
    """
    cluster the users by mean shift
    :param file_path:
    :param bandwidth:
    :return: label list sorted by uid
    """
    matrix = fp.get_data2_matrix(file_path)

    train_data = []
    plt_x = []
    ply_y = []

    labels = {}
    m, n = matrix.shape

    for uid in range(m):
        for iid in range(n):
            if matrix[uid, iid] != 0:
                plt_x.append(uid)
                ply_y.append(matrix[uid, iid])
                train_data.append([uid, matrix[uid, iid]])

    clf = MeanShift(bandwidth=bandwidth).fit(train_data)

    for i in range(len(plt_x)):
        if plt_x[i] not in labels:
            labels[plt_x[i]] = clf.labels_[i]

    return labels


def add_label_matrix(lables, matrix, uid_indices, mapping):
    """
    put the cluster label into matrix as feature
    :param lables:
    :param matrix:
    :return:
    """
    matrix = np.array(matrix)
    m, n = matrix.shape
    new_matrix = np.zeros((m, n + 1))

    for i in range(n):
        new_matrix[:, i] = matrix[:, i]

    for i in range(m):
        id = int(matrix[i, uid_indices])
        if id in mapping:
            new_matrix[i, n] = lables[mapping[id]]
        else:
            new_matrix[i, n] = 0

    return new_matrix


if __name__ == '__main__':
    item_cluster_show()
