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


def similar_user_cluster():
    """
    cluster by similar user
    :return:
    """
    cluster = []
    collected_user = set()
    collecting_user = set()
    file_prefix = "resources/data/collaborative/"

    users = get_similar_by_read_file(file_prefix + "0.txt")
    collected_user.add(0)
    for v in users:
        collecting_user.add(v)

    flag = np.zeros(13758)
    flag[0] = 1

    while True:
        # find a cluster
        while len(collecting_user) != 0:
            another = set()
            for v in collecting_user:
                if v not in collected_user and flag[v]==0:
                    flag[v] = 1
                    collected_user.add(v)

                    current_users = get_similar_by_read_file(file_prefix + str(v) + ".txt")
                    for value in current_users:
                        another.add(value)

            collecting_user = another.copy()

        cluster.append(collected_user)

        rest = np.where(flag == 0)
        rest=rest[0]
        print(len(rest))
        if len(rest) == 0:
            break

        collected_user.clear()
        collecting_user.clear()
        current_id = rest[0]
        users = get_similar_by_read_file(file_prefix + str(current_id) + ".txt")
        collected_user.add(current_id)
        for v in users:
            collecting_user.add(v)

        flag[current_id] = 1

    return len(cluster)


def get_similar_by_read_file(file_name):
    """
    get similar users by read the file
    :param file_name:
    :return:
    """
    similar_user = []
    f = open(file_name, 'r')
    for lines in f.readlines():
        id = int(lines.split('\t')[0])
        similar_user.append(id)

    f.close()
    return similar_user


if __name__ == '__main__':
    print(similar_user_cluster())
