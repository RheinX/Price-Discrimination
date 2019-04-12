import regression.l1_gsd as lgsd
import regression.diferent_regression as df
import matplotlib.pyplot as plt
import regression.curve_fit as cf
import error_handle as eh
import cluster.cluster as cl
import feature_prepare as fp

def multiple_feature_regression():
    """
    # regression predict by using all data with different features
    :return:
    """

    # read the data
    dataMatrix, labelMatrix = lgsd.load_data("../resources/data/TrainingSet.csv")
    test_dataMatrix, test_labelMatrix = lgsd.load_data("../resources/data/TestSet.csv")

    # cluster
    user_mapping = fp.get_mapping("../resources/data/clean_data_2/user_mapping.txt")

    train_labels_cluster = cl.meanshift_cluser("../resources/data/clean_data_2/", 3)
    test_labels_cluster = cl.meanshift_cluser("../resources/data/test_data_2/", 3)

    dataMatrix = cl.add_label_matrix(train_labels_cluster, dataMatrix, 6, user_mapping)
    test_dataMatrix = cl.add_label_matrix(test_labels_cluster, test_dataMatrix, 6, user_mapping)

    # extract the feature
    weight = lgsd.feature_select(dataMatrix, labelMatrix)
    dataMatrix = lgsd.remove_feature(dataMatrix, weight)
    test_dataMatrix = lgsd.remove_feature(test_dataMatrix, weight)



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
    br_rate ,rate= df.BayesianRidge_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)
    plt_x = []
    plt_y = []
    for v in br_rate:
        if float(v) <= 100:
            plt_x.append(float(v))
            plt_y.append(round(float(br_rate[v]), 4))

    print(rate)
    plt.title("Predict in BayesianRidge")
    plt.xlabel("error with real value")
    plt.ylabel("Percentage of total")
    plt.scatter(plt_x, plt_y)
    plt.show()

def one_dimension_fitting():
    """
    fit the curve which data is price of one user to one category item
    :return:
    """
    error, right_rate=cf.polynomial_regression(3)
    plt_x,plt_y=eh.plt_array(error,100)

    print(right_rate)
    plt.title("Predict in polynomial Regression,dimension:3")
    plt.xlabel("error with real value")
    plt.ylabel("Percentage of total")
    plt.scatter(plt_x, plt_y)
    plt.show()


if __name__ == '__main__':
    # regression predict by using all data with different features
    # multiple_feature_regression()
    multiple_feature_regression()