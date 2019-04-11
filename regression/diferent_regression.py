from sklearn import svm
import error_handle as eh
from sklearn.linear_model import SGDRegressor as gsd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge

def svm_regression(dataMatrix, labelMatrix, test_dataMatirx, test_labelMatrix):
    """
    use svm to predict
    :param dataMatrix:
    :param labelMatrix:
    :param test_dataMatirx:
    :param test_labelMatrix:
    :return:
    """
    clf = svm.SVR()
    clf.fit(dataMatrix,labelMatrix)
    predict=clf.predict(test_dataMatirx)

    error_rate=eh.error_rate(predict,test_labelMatrix)
    print(error_rate)

def gsd_regression(dataMatrix, labelMatrix, test_dataMatirx, test_labelMatrix):
    clf = gsd()
    clf.fit(dataMatrix, labelMatrix)
    predict = clf.predict(test_dataMatirx)

    error_rate = eh.error_rate(predict, test_labelMatrix)
    print(error_rate)

def LogisticRegression(dataMatrix, labelMatrix, test_dataMatirx, test_labelMatrix):
    clf=lr()
    clf.fit(dataMatrix,labelMatrix)
    predict = clf.predict(test_dataMatirx)

    error_rate = eh.error_rate(predict, test_labelMatrix)
    print(error_rate)

def LassoLars_regression(dataMatrix, labelMatrix, test_dataMatirx, test_labelMatrix):
    clf=LassoLars()
    clf.fit(dataMatrix, labelMatrix)
    predict = clf.predict(test_dataMatirx)

    error_rate = eh.error_rate(predict, test_labelMatrix)
    print(error_rate)

def BayesianRidge_regression(dataMatrix, labelMatrix, test_dataMatirx, test_labelMatrix):
    clf=BayesianRidge()
    clf.fit(dataMatrix, labelMatrix)
    predict = clf.predict(test_dataMatirx)

    error_rate = eh.error_rate(predict, test_labelMatrix,0.1)
    return error_rate