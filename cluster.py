import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import feature_prepare as fp


def user_cluster_show():
    """
    use user_item matrix to show the situation of cluster
    :return:
    """
    matrix = fp.get_data2_matrix("resources/data/clean_data_2/")
    plt_x = []
    ply_y = []

    m, n = matrix.shape

    for uid in range(m):
        for iid in range(n):
            if matrix[uid, iid] != 0:
                plt_x.append(uid)
                ply_y.append(matrix[uid, iid])

    plt.scatter(plt_x,ply_y)
    plt.show()


if __name__ == '__main__':
    matrix = fp.get_data2_matrix("resources/data/clean_data_2/")

    train_data=[]
    plt_x = []
    ply_y = []

    m, n = matrix.shape

    for uid in range(m):
        for iid in range(n):
            if matrix[uid, iid] != 0:
                plt_x.append(uid)
                ply_y.append(matrix[uid, iid])
                train_data.append([uid,matrix[uid, iid]])

    clf=MeanShift(bandwidth=20).fit(train_data)
    lables=clf.labels_

    lables=set(lables)
    print(len(lables))