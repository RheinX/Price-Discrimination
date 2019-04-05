import regression.l1_gsd as lgsd
import regression.diferent_regression as df
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read the data
    dataMatrix, labelMatrix = lgsd.load_data("../resources/data/TrainingSet.csv")
    test_dataMatrix, test_labelMatrix = lgsd.load_data("../resources/data/TestSet.csv")

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
    br_rate = df.BayesianRidge_regression(dataMatrix, labelMatrix, test_dataMatrix, test_labelMatrix)
    plt_x = []
    plt_y = []
    for v in br_rate:
        if float(v) <= 100:
            plt_x.append(float(v))
            plt_y.append(round(float(br_rate[v]), 4))

    plt.title("Predict in BayesianRidge")
    plt.xlabel("error with real value")
    plt.ylabel("Percentage of total")
    plt.scatter(plt_x, plt_y)
    plt.show()
