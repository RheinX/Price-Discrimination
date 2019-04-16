import regression.l1_gsd as lgsd
import regression.diferent_regression as df
import matplotlib.pyplot as plt
import regression.curve_fit as cf
import error_handle as eh
import cluster.cluster as cl
import feature_prepare as fp


def load_feature(train_path, test_path):
    """
    read the data and return train and test data
    the data has been normalized and remove useless feature
    :return:
    """
    # read the data
    dataMatrix, labelMatrix = lgsd.load_data(train_path)
    test_dataMatrix, test_labelMatrix = lgsd.load_data(test_path)

    # extract the feature
    weight = lgsd.feature_select(dataMatrix, labelMatrix)
    print(weight)
    dataMatrix = lgsd.remove_feature(dataMatrix, weight)
    test_dataMatrix = lgsd.remove_feature(test_dataMatrix, weight)

    return dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix

def multiple_feature_regression_cluster(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix, size):
    """

    :param dataMatrix:
    :param labelMatrix:
    :param test_dataMatrix:
    :param test_labelMatrix:
    :param size: the size of window of meanshift
    :return:
    """
    # cluster
    user_mapping = fp.get_mapping("../resources/data/clean_data_2/user_mapping.txt")

    train_labels_cluster = cl.meanshift_cluser("../resources/data/clean_data_2/", size)
    test_labels_cluster = cl.meanshift_cluser("../resources/data/test_data_2/", size)

    dataMatrix = cl.add_label_matrix(train_labels_cluster, dataMatrix, 6, user_mapping)
    test_dataMatrix = cl.add_label_matrix(test_labels_cluster, test_dataMatrix, 6, user_mapping)

    # ok
    print("Window size:"+str(size)+" Start")
    br_rate = df.BayesianRidge_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)

    plt_x, plt_y = eh.plt_sum(br_rate)
    # plt.title("Predict in data feature")
    # plt.xlabel("error rate")
    # plt.ylabel("error number")
    # plt.plot(plt_x, plt_y)
    # plt.show()

    return plt_x, plt_y

def multiple_feature_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix):
    """
    # regression predict by using all data with different features
    :return:
    """
    # predict price by using different Algorithms

    # the result of this Algorithm is wrong
    # print("GSD Start!")
    # df.gsd_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)

    # all predict is same
    # print("SVM Start!")
    # df.svm_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)

    # wrong
    # print("Logistic Regression start!")
    # df.LogisticRegression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)

    # all predict is same
    # print("LassoLars Start")
    # df.LassoLars_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)

    # ok
    print("BayesianRidge Start")
    br_rate = df.BayesianRidge_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)
    # plt_x = []
    # plt_y = []
    # for v in br_rate:
    #     if float(v) <= 100:
    #         plt_x.append(float(v))
    #         plt_y.append(round(float(br_rate[v]), 4))
    #
    # print(rate)
    plt_x, plt_y = eh.plt_sum(br_rate)
    # plt.title("Predict in data feature")
    # plt.xlabel("error rate")
    # plt.ylabel("error number")
    # plt.plot(plt_x, plt_y)
    # plt.show()

    return plt_x, plt_y


def one_dimension_fitting(pol_dem):
    """
    fit the curve which data is price of one user to one category item
    :return:
    """
    error = cf.polynomial_regression(pol_dem)
    plt_x, plt_y = eh.plt_sum(error)

    # print(right_rate)
    # plt.title("Predict in polynomial Regression,dimension:3")
    # plt.xlabel("error with real value")
    # plt.ylabel("Percentage of total")
    # plt.plot(plt_x, plt_y)
    # plt.show()
    return plt_x,plt_y


if __name__ == '__main__':
    # regression predict by using all data with different features
    dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix\
        =load_feature("../resources/data/TrainingSet.csv","../resources/data/TestSet.csv")
    plt_x_regression, plt_y_regression=\
        multiple_feature_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)
    # plt_x_cluster3,plt_y_cluster3=\
    #     multiple_feature_regression_cluster(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix,3)
    # plt_x_cluster5, plt_y_cluster5 = \
    #     multiple_feature_regression_cluster(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix, 5)
    # plt_x_cluster10, plt_y_cluster10 = \
    #     multiple_feature_regression_cluster(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix, 10)
    # plt_x_cluster20, plt_y_cluster20 = \
    #     multiple_feature_regression_cluster(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix, 20)

    plt.title("Predict in data feature")
    plt.xlabel("error rate")
    plt.ylabel("error number")

    plt.plot(plt_x_regression, plt_y_regression)
    # plt.plot(plt_x_regression,plt_y_regression,color="green",label="no cluster")
    # plt.plot(plt_x_cluster3, plt_y_cluster3, color="blue", label="window size:3")
    # plt.plot(plt_x_cluster5, plt_y_cluster5, color="red", label="window size:5")
    # plt.plot(plt_x_cluster10, plt_y_cluster10, color="skyblue", label="window size:10")
    # plt.plot(plt_x_cluster20, plt_y_cluster20, color="black", label="window size:20")

    plt.legend()
    plt.show()

    # # curve fit
    # plt_x_regression,plt_y_regression=one_dimension_fitting(1)
    # plt_x2, plt_y2 = one_dimension_fitting(2)
    # plt_x3, plt_y3 = one_dimension_fitting(3)
    #
    # plt.title("Predict in curve fitting")
    # plt.xlabel("error rate")
    # plt.ylabel("error percent")
    #
    # plt.plot(plt_x_regression,plt_y_regression,color="blue",label="linear")
    # plt.plot(plt_x2, plt_y2, color="red", label="Binary polynomial")
    # plt.plot(plt_x3, plt_y3, color="green", label="Ternary polynomial")
    #
    # plt.legend()
    # plt.show()