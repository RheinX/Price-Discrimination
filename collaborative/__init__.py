import collaborative.based_user as bu
import error_handle as eh
import matplotlib.pyplot as plt
import feature_prepare as fp
import smote

from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge


def filter_based_formula(train_file_path, test_file_path):
    """
    predict the pirce of formula
    :param train_file_path:
    :param test_file_path:
    :param clf:
    :return:
    """
    similarity_path = "../resources/data/collaborative/"
    # write similarity user files
    # bu.get_similarity_users(train_file_path,similarity_path)

    # read the matrix of similarity
    similarity = bu.read_similarity(similarity_path)

    # get the train and test matrix
    train_matrix = fp.get_data2_matrix(train_file_path)
    test_matrix = fp.get_data2_matrix(test_file_path)

    label = []
    predict = []
    m, n = test_matrix.shape

    # this method predict the price by the formula
    for uid in range(m):
        for iid in range(n):
            if test_matrix[uid, iid] != 0:
                prices = bu.predict_based_formula(similarity, train_matrix, uid, iid)

                if 0 != prices:
                    label.append(test_matrix[uid, iid])
                    predict.append(prices)

    # error, rate = eh.error_rate(predict, label, 0.1)
    error = eh.error_rate_sum(predict, label, 0.3)
    plt_x, plt_y = eh.plt_sum(error)

    # # #print(rate)
    # plt.title("Predict in formula")
    # plt.xlabel("error rate")
    # plt.ylabel("number percentage")
    # plt.plot(plt_x, plt_y)
    # plt.show()

    return plt_x, plt_y


def filter_based_user(train_file_path, test_file_path, clf,num):
    """
    algorithm like collaborative filter based on users
    :param train_file:
    :param test_file:
    :return:
    """
    similarity_path = "../resources/data/collaborative/"
    # write similarity user files
    # bu.get_similarity_users(train_file_path,similarity_path)

    # read the matrix of similarity
    similarity = bu.read_similarity(similarity_path)

    # get the train and test matrix
    train_matrix = fp.get_data2_matrix(train_file_path)
    test_matrix = fp.get_data2_matrix(test_file_path)

    train_matrix_svd = fp.get_svd_matrix("../resources/data/svd/nb.txt")
    test_matrix_svd = fp.get_svd_matrix("../resources/data/svd/nb-test.txt")

    label = []
    predict = []
    m, n = test_matrix.shape

    # # regression
    for uid in range(m):
        train_data, train_label, train_item_mapping, train_user_mapping = bu.produce_data_to_train(uid, train_matrix,
                                                                                                   similarity)

        # lower the true rate,fuck!
        # # normalize
        # scaler = MinMaxScaler()
        # scaler.fit(train_data)
        # train_data = scaler.transform(train_data)

        test_data, test_label, test_item_mapping, test_user_mapping = bu.produce_data_to_train(uid, test_matrix,
                                                                                               similarity)

        # fill the test_data
        train_data, train_label = bu.fill_matrix(train_data, train_label, train_item_mapping, train_user_mapping,
                                                 train_matrix_svd)
        test_data, test_label = bu.fill_matrix(test_data, test_label, test_item_mapping, test_user_mapping,
                                               test_matrix_svd)

        if len(test_data) == 0:
            continue

        # regression
        # clf=SGDRegressor()
        # clf = svm.SVR()
        # clf=LassoLars()
        # clf=BayesianRidge()

        # extend the data by SMOTE
        if 0!=num:
            train_data, train_label = smote.fill_matrix(train_data, train_label, num)

        clf.fit(train_data, train_label)
        result = clf.predict(test_data)

        # if len(test_label)!=len(result):
        #     print("fuck")
        #     print(len(test_label))
        #     print(len(result))
        #     print(len(test_data))
        #     print()
        label.extend(test_label)
        predict.extend(result)

    # error, rate = eh.error_rate(predict, label, 0.1)
    error = eh.error_rate_sum(predict, label, 0.3)
    plt_x, plt_y = eh.plt_sum(error)

    return plt_x, plt_y
    # #print(rate)
    # plt.title("PFCF model")
    # plt.xlabel("error rate")
    # plt.ylabel("number of error")
    # plt.plot(plt_x, plt_y)
    # plt.show()


if __name__ == '__main__':
    train_file_path = "../resources/data/clean_data_2/"
    test_file_path = "../resources/data/test_data_2/"

    # # regression
    # clf_svm = svm.SVR()
    # clf_sgd = SGDRegressor()
    # clf_lass = LassoLars()
    # clf_ridge = BayesianRidge()
    #
    # plt_x_svm, plt_y_svm = filter_based_user(train_file_path, test_file_path, clf_svm)
    # plt_x_sgd, plt_y_sgd = filter_based_user(train_file_path, test_file_path, clf_sgd)
    # plt_x_lass, plt_y_lass = filter_based_user(train_file_path, test_file_path, clf_lass)
    #plt_x_ridge, plt_y_ridge = filter_based_user(train_file_path, test_file_path, clf_ridge)
    #

    # # # formula
    # plt_x_formula, plt_y_formula=filter_based_formula(train_file_path,test_file_path)

    # # smote
    clf = svm.SVR()
    # clf = SGDRegressor()
    # clf = LassoLars()
    # clf = BayesianRidge()
    plt_x, plt_y=filter_based_user(train_file_path,test_file_path,clf,0)
    plt_x1, plt_y1 = filter_based_user(train_file_path, test_file_path, clf, 100)
    plt_x2, plt_y2 = filter_based_user(train_file_path, test_file_path, clf, 200)
    plt_x3, plt_y3 = filter_based_user(train_file_path, test_file_path, clf, 300)
    plt_x4, plt_y4 = filter_based_user(train_file_path, test_file_path, clf, 400)
    plt_x5, plt_y5 = filter_based_user(train_file_path, test_file_path, clf, 500)

    # # draw
    plt.title("Predict in SMOTE and SVM")
    plt.xlabel("error rate")
    plt.ylabel("error percentage")
    #
    # plt.plot(plt_x_svm, plt_y_svm, color="green", label="svm")
    # plt.plot(plt_x_sgd, plt_y_sgd, color="blue", label="sgd")
    # plt.plot(plt_x_lass, plt_y_lass, color="red", label="LassoLars")
    # plt.plot(plt_x_formula, plt_y_formula, color="red", label="formula")
    # plt.plot(plt_x_ridge, plt_y_ridge, color="skyblue", label="BayesianRidge")

    plt.plot(plt_x,plt_y,color="green",label="origin data")
    plt.plot(plt_x1, plt_y1, color="red", label="100")
    plt.plot(plt_x2, plt_y2, color="black", label="200")
    plt.plot(plt_x3, plt_y3, color="blue", label="300")
    plt.plot(plt_x4, plt_y4, color="skyblue", label="400")
    plt.plot(plt_x5, plt_y5, color="gray", label="500")
    plt.legend()
    plt.show()

