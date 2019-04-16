import csv
import numpy as np
from sklearn.linear_model import SGDRegressor as gsd
from sklearn.preprocessing import MinMaxScaler


def load_data(fileName):
    """
    import data from file data_2, return the dataMarix and labelMatrix
    :param fileName:
    :return:
    """
    dataMatrix = []
    labelMatrix = []

    # read data
    no_need_feature = ["EbayID", "Price", "SellerName", "EndDay"]  # the feature do not need to push into dataMatrix
    with open(fileName) as f:
        f_csv = csv.DictReader(f)
        # read label
        # read data which type is not str
        for row in f_csv:
            labelMatrix.append(float(row["Price"]))

            row_data = []
            for feature in row:
                if feature not in no_need_feature:
                    row_data.append(float(row[feature]))

            dataMatrix.append(row_data)

    return dataMatrix, labelMatrix


def feature_select(dataMatrix, labelMatrix):
    """
    extract feature by gsd and return the weight
    :param dataMatrix:
    :param labelMatrix:
    :return:
    """
    # normalize the data
    scaler = MinMaxScaler()
    scaler.fit(dataMatrix)
    dataMatrix = scaler.transform(dataMatrix)

    # gsd based l1
    clf = gsd(penalty="l2")
    clf.fit(dataMatrix, labelMatrix)
    return clf.coef_


def remove_feature(dataMatrix, weight):
    """
    delete the column which corresponding weight is 0
    :param dataMatrix:
    :param weight:
    :return:
    """

    deleteIndices = np.where(weight == 0)
    dataMatrix = np.delete(dataMatrix, deleteIndices, axis=1)
    return dataMatrix
