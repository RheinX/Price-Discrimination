import collaborative.based_user as bu
import error_handle as eh
import matplotlib.pyplot as plt
import feature_prepare as fp
import smote

from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler

def filter_based_user(train_file_path, test_file_path):
    """
    algorithm like collaborative filter based on users
    :param train_file:
    :param test_file:
    :return:
    """
    similarity_path="../resources/data/collaborative/"
    # write similarity user files
    # bu.get_similarity_users(train_file_path,similarity_path)

    # read the matrix of similarity
    similarity=bu.read_similarity(similarity_path)

    # get the train and test matrix
    train_matrix=fp.get_data2_matrix(train_file_path)

    test_matrix=fp.get_data2_matrix(test_file_path)

    label=[]
    predict=[]
    m,n=test_matrix.shape

    # # this method predict the price by the formula
    # for uid in range(m):
    #     for iid in range(n):
    #         if test_matrix[uid,iid]!=0:
    #             prices=bu.predict_based_formula(similarity,train_matrix,uid,iid)
    #
    #             if 0!=prices:
    #                 label.append(test_matrix[uid,iid])
    #                 predict.append(prices)

    # # regression
    for uid in range(m):
        train_data,train_label=bu.produce_data_to_train(uid,train_matrix,similarity)

        # lower the true rate,fuck!
        # # normalize
        # scaler = MinMaxScaler()
        # scaler.fit(train_data)
        # train_data = scaler.transform(train_data)

        test_data,test_label=bu.produce_data_to_train(uid,test_matrix,similarity)

        # fill the test_data
        test_data,test_label=bu.fill_matrix(test_data,test_label)

        if len(test_data)==0:
            continue

        # regression
        # clf=SGDRegressor()
        clf=svm.SVR()
        # clf=LassoLars()
        # clf=BayesianRidge()

        # extend the data by SMOTE
        train_data, train_label= smote.fill_matrix(train_data,train_label,500)

        clf.fit(train_data,train_label)
        result=clf.predict(test_data)

        # if len(test_label)!=len(result):
        #     print("fuck")
        #     print(len(test_label))
        #     print(len(result))
        #     print(len(test_data))
        #     print()
        label.extend(test_label)
        predict.extend(result)

    error,rate=eh.error_rate(predict,label,0.1)
    plt_x,plt_y=eh.plt_array(error)

    print(rate)
    # plt.title("Predict in SVM regression")
    # plt.xlabel("error with real value")
    # plt.ylabel("Percentage of total")
    # plt.scatter(plt_x, plt_y)
    # plt.show()


if __name__ == '__main__':
    train_file_path="../resources/data/clean_data_2/"
    test_file_path="../resources/data/test_data_2/"
    filter_based_user(train_file_path,test_file_path)
