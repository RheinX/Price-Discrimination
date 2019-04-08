import feature_prepare as fp
from sklearn.linear_model import LinearRegression as lr
<<<<<<< HEAD
from sklearn.preprocessing import PolynomialFeatures
import error_handle as eh


=======
import error_handle as eh

>>>>>>> origin/master
def linear_regression():
    """
    predict by linear predict
    :param data:
    :return:
    """
    # get the predict
    file_prefix = "../resources/data/clean_data_2/user/"
<<<<<<< HEAD
    user = fp.get_user_item_log(file_prefix)
    user_predict = {}
    for uid in user:
        user_predict[uid] = {}
        for iid in user[uid]:
            prices = user[uid][iid]
            x, y = produce_test_data(prices)
            clf = lr()
            clf.fit(x, y)
            predict_data = clf.predict([[len(x)]])
            user_predict[uid][iid] = predict_data

    # get the test data
    file_prefix = "../resources/data/test_data_2/"
    test_matrix = fp.get_data2_matrix(file_prefix)  # get the matrix which store the max price of one user to one item
    label = []
    pre_data = []

    m, n = test_matrix.shape
    for uid in range(m):
        for iid in range(n):
            if test_matrix[uid, iid] != 0 and str(iid) in user_predict[str(uid)]:
                label.append(test_matrix[uid, iid])
                pre_data.append(user_predict[str(uid)][str(iid)][0])

    result = eh.error_rate(pre_data, label, 0.3)
    return result


def polynomial_regression(degree):
    """
    predict by polynomial regression
    :return:
    """
    """
    predict by linear predict
    :param data:
    :return:
    """
    # get the predict
    file_prefix = "../resources/data/clean_data_2/user/"
    user = fp.get_user_item_log(file_prefix)
    user_predict = {}
    for uid in user:
        user_predict[uid] = {}
        for iid in user[uid]:
            prices = user[uid][iid]
            x, y = produce_test_data(prices)

            quadratic_featurizer = PolynomialFeatures(degree=degree)
            x_train_quadratic = quadratic_featurizer.fit_transform(x)

            clf = lr()
            clf.fit(x_train_quadratic, y)
            test_data=quadratic_featurizer.transform([[len(x)]])
            predict_data = clf.predict(test_data)
            user_predict[uid][iid] = predict_data

    # get the test data
    file_prefix = "../resources/data/test_data_2/"
    test_matrix = fp.get_data2_matrix(file_prefix)  # get the matrix which store the max price of one user to one item
    label = []
    pre_data = []

    m, n = test_matrix.shape
    for uid in range(m):
        for iid in range(n):
            if test_matrix[uid, iid] != 0 and str(iid) in user_predict[str(uid)]:
                label.append(test_matrix[uid, iid])
                pre_data.append(user_predict[str(uid)][str(iid)][0])

    result = eh.error_rate(pre_data, label, 0.3)
    return result


=======
    user=fp.get_user_item_log(file_prefix)
    user_predict={}
    for uid in user:
        user_predict[uid]={}
        for iid in user[uid]:
            prices=user[uid][iid]
            x, y=produce_test_data(prices)
            clf=lr()
            clf.fit(x,y)
            predict_data=clf.predict([[len(x)]])
            user_predict[uid][iid]=predict_data

    # get the test data
    file_prefix = "../resources/data/test_data_2/"
    test_matrix=fp.get_data2_matrix(file_prefix)  # get the matrix which store the max price of one user to one item
    label=[]
    pre_data=[]

    m,n=test_matrix.shape
    for uid in range(m):
        for iid in range(n):
            if test_matrix[uid,iid]!=0 and str(iid) in user_predict[str(uid)]:
                label.append(test_matrix[uid,iid])
                pre_data.append(user_predict[str(uid)][str(iid)][0])

    result=eh.error_rate(pre_data,label,0.3)
    return result

>>>>>>> origin/master
def produce_test_data(data):
    """
    produce data x of 2-d and y of 1-d by a list
    :param data:
    :return:
    """
<<<<<<< HEAD
    x = []
    y = []
=======
    x=[]
    y=[]
>>>>>>> origin/master
    for value in data:
        x.append([len(x)])
        y.append(value)

    return x, y
<<<<<<< HEAD
=======

def predict():
    file_prefix = "../resources/data/clean_data_2/user/"
    user=fp.get_user_item_log(file_prefix)
    user_predict={}
    for uid in user:
        user_predict[uid]={}
        for iid in user[uid]:
            prices=user[uid][iid]
            x, y=produce_test_data(prices)
            clf=lr()
            clf.fit(x,y)
            predict_data=clf.predict([[len(x)]])
            user_predict[uid][iid]=predict_data

    return user_predict
>>>>>>> origin/master
