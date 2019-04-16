import feature_prepare as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def matrix_handle(vector, matrix):
    """
    delete the column which value in of vecotor is 0 of matrix
    :param vector: vector of target user
    :param matrix:
    :return:
    """
    delete_column = np.where(vector == 0)[0]
    matrix_spare = matrix.copy()

    m, n = matrix.shape
    for column in delete_column:
        matrix_spare[:, column] = np.zeros(m)

    return matrix_spare


# todo: change the formula of similarity
def get_similarity(vector, matrix, uid, num=100):
    """
    calculate the similarity between vector and each vector in matrix
    in this method, we use Manhattan Distance
    return a tuple sorted by similarity: key is user id and value is similarity

    :param vector:
    :param matrix:
    :param uid: the id of vector, we need to remove this when return the similarity dict
    :param num: the number of similar user need to save
    :return:
    """
    m, n = matrix.shape
    diff_matrix = np.tile(vector, (m, 1)) - matrix
    diff_matrix = abs(diff_matrix)

    distance = diff_matrix.sum(axis=1)

    # put the id and similarity into a dict
    similarity = {}
    for id in range(m):
        if id != uid:
            similarity[id] = distance[id]

    # sort
    similarity_sorted = sorted(similarity.items(), key=lambda items: items[1])
    similarity = similarity_sorted[0:num]

    return similarity


def get_similarity_users(train_file_path, save_path):
    """
    get some similar users of target user and write the id and similarity into file which path is save_path
    :param train_file_path:
    :param save_path:
    :return:
    """
    # prepare training data
    dataMatrix = fp.get_data2_matrix(train_file_path)

    # normalize the matrix
    scaler = MinMaxScaler()
    scaler.fit(dataMatrix)
    dataMatrix = scaler.transform(dataMatrix)

    m, n = dataMatrix.shape
    for uid in range(m):
        for iid in range(n):
            target_vector = dataMatrix[uid, :]
            matrix_spare = matrix_handle(target_vector, dataMatrix)

            # get the tuple of similarity
            similarity = get_similarity(target_vector, matrix_spare, uid)

            # save the tuple split with \t: id,similarity
            f = open(save_path + str(uid) + ".txt", 'w')
            for k, v in similarity:
                f.write(str(k) + '\t' + str(v) + '\n')
            f.close()


def read_similarity(file_path):
    """
    read similarity from file into a dict
    :param file_path:
    :return:
    """
    file_names = os.listdir(file_path)

    similarity = {}
    for names in file_names:
        uid = names.split('.')[0]
        uid = int(uid)

        similarity[uid] = {}
        f = open(file_path + names, 'r')
        for lines in f.readlines():
            lines = lines.split('\t')
            id = int(lines[0])
            eff = float(lines[1])

            similarity[uid][id] = eff

        f.close()

    return similarity


def predict_based_formula(similarity, train_matrix, uid, iid):
    """
    predict the price whose is iid of user whose id is uid by formula of similarity
    :param similarity:
    :param train_matrix:
    :param uid:
    :param iid:
    :return:
    """
    simi_users = similarity[uid]

    prices = []  # store the prices of similary user
    coefficient = []  # store the coefficient of similarity

    for id in simi_users:
        if train_matrix[id, iid] != 0:
            prices.append(train_matrix[id, iid])
            coefficient.append(simi_users[id])

    if 0 == len(prices):
        return 0

    avg_prices = sum(prices) / len(prices)
    prices = np.array(prices)
    total_coefficient = sum(coefficient)
    coefficient = np.array(coefficient)
    coefficient = coefficient / total_coefficient

    avg_p = np.full(len(prices), avg_prices*0.75)

    predict_price = avg_prices + sum(np.multiply(coefficient, avg_p - prices))

    return predict_price


# todo :fill the matrix with SVD
def produce_data_to_train(uid, matrix, similarity):
    """
    return a matrix which x are similar users and y is item
    :param uid:
    :param matrix:
    :return:
    """
    item_mapping = {}
    user_mapping = {}

    label = []
    simi_users = similarity[uid]
    item_index = np.where(matrix[uid, :] != 0)[0]  # item id that this user(uid) aucte
    train_data = np.zeros((len(item_index), len(simi_users)))

    # fill the matrix

    for iid in range(len(item_index)):
        item_mapping[iid] = item_index[iid]
        i = 0
        label.append(matrix[uid, item_index[iid]])
        for id in simi_users:
            user_mapping[i] = id
            train_data[iid, i] = matrix[id, item_index[iid]]
            i += 1

    return train_data, label, item_mapping, user_mapping


def fill_matrix(matrix, label, item_mapping, user_mapping,matrix_svd):
    """
    fill the element equal 0
    the principle of proximity, delete this row if all element is 0
    :param matrix:
    :return:
    """

    # # fill nearest element
    # delete_row=[]
    # m,n=matrix.shape
    # for i in range(m):
    #     vector=matrix[i,:]
    #     d_vector=np.where(vector==0)[0]
    #     if len(d_vector)==len(vector):
    #         delete_row.append(i)
    #         continue
    #
    #     p=0
    #     while matrix[i,p]==0:
    #         p+=1
    #
    #     current = matrix[i, p]
    #     for j in range(n):
    #         if matrix[i,j]==0:
    #             matrix[i,j]=current
    #         else:
    #             current=matrix[i,j]
    #
    # # delete the row
    # matrix=np.delete(matrix,delete_row,axis=0)
    # label=np.delete(label,delete_row)

    # # fill with svd
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i,j]==0:
                matrix[i,j]=matrix_svd[user_mapping[j],item_mapping[i]]

    return matrix, label
