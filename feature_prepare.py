import numpy as np
import os


def get_data2_matrix(file_prefix):
    """
    read the price of a user to a item into a matrix
    :return:
    """
    # file_prefix="resources/data/clean_data_2/"

    # read the size of item and user
    user_item_name = file_prefix + "user_item.txt"
    user_item_file = open(user_item_name, 'r')
    lines = user_item_file.readlines()[0]
    lines = lines.split('\t')
    item_num = int(lines[0])
    user_num = int(lines[1])

    user_item_matrix = np.zeros((user_num, item_num))  # the matrix

    # fill the matrix
    user_file_name = file_prefix + "user/"
    files_name = os.listdir(user_file_name)
    for name in files_name:
        uid = int(name.split('_')[0])
        file = open(user_file_name + name)
        for line in file.readlines():
            line = line.split('\t')
            iid = int(line[0])
            prices = line[4:]
            prices = [float(i) for i in prices]
            max_price = max(prices)
            user_item_matrix[uid, iid] = max_price
        file.close()

    return user_item_matrix


def get_user_item_log(file_prefix):
    """
    load the history data price of user

    :return:
    """
    user = {}
    # loop the file
    name_lists = os.listdir(file_prefix)
    for file_name in name_lists:
        user_id = file_name.split('_')[0]

        user[user_id] = {}
        f = open(file_prefix + file_name)
        for line in f.readlines():
            line = line.split('\t')
            category_id = line[0]
            prices = line[4:]
            prices = [float(i) for i in prices]

            user[user_id][category_id] = prices
        f.close()

    return user


def get_item_price(file_path):
    """
    get the item list and their avg price
    :param file_path:
    :return:
    """
    item = {}

    f = open(file_path, 'r')
    for lines in f.readlines():
        lines = lines.split('\t')

        iid = int(lines[0])
        avg = float(lines[2])

        item[iid] = avg

    return item


def get_mapping(file_name):
    """
    get the mappping and return a dict
    :param file_name:
    :return:
    """
    mapping = {}
    f = open(file_name, 'r')
    for lines in f.readlines():
        lines = lines.split('\t')
        id = int(lines[0])
        uid = int(lines[1])

        mapping[id] = uid

    return mapping


def get_svd_matrix(file_name):
    matrix = []
    f = open(file_name, 'r')

    for lines in f.readlines():
        lines = lines.split('\t')
        lines = [float(x) for x in lines]
        matrix.append(lines)

    f.close()

    return np.array(matrix)
