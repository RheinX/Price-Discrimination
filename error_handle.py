import numpy as np


def error_rate_sum(predict, label, threshold):
    """
    calculate the offset between predict and label
    return their result for drawing
    :param predict:
    :param label:
    :param threshold: the max error rate of pic
    :return:
    """
    predict = np.array(predict)
    label = np.array(label)
    error = abs(predict - label)

    size = len(error)
    result = {}
    for i in range(size):
        v = float(error[i]) / float(label[i])

        if v > threshold:
            continue

        offset = round(v, 2)

        if offset not in result:
            result[offset] = 0

        result[offset] += 1

    result_sort = sorted(result.items())
    error_plt={}
    error_plt[result_sort[0][0]]=result_sort[0][1]
    for i in range(1, len(result_sort)):
        error_plt[result_sort[i][0]] = result_sort[i][1]+error_plt[result_sort[i - 1][0]]

    return sorted(error_plt.items())


def plt_sum(data):
    plt_x = []
    plt_y = []

    for i in range(len(data)):
        plt_x.append(data[i][0])
        plt_y.append(data[i][1])

    return plt_x, plt_y


def error_rate(predict, label, rate):
    """
    calculate the offset between predict and label
    return their result and right rate
    :param predict:
    :param label:
    :param rate:
    :return:
    """
    predict = np.array(predict)
    label = np.array(label)
    error = abs(predict - label)

    right_rate = 0.0
    size = len(error)
    result = {}
    for i in range(size):
        v = float(error[i]) / float(label[i])

        offset = str(round(v, 2))

        if offset not in result:
            result[offset] = 0

        result[offset] += 1

    for v in result:
        num = result[v]
        result[v] = float(result[v]) / size

        if float(v) <= rate:
            right_rate += num

    return result, float(right_rate) / size


def plt_array(data, rate=100, filter=True):
    """
    use filter to control data
    :param data:
    :param rate:
    :param filter:
    :return:
    """
    plt_x = []
    plt_y = []
    for v in data:
        if filter:
            if float(v) <= rate:
                plt_x.append(float(v))
                plt_y.append(round(float(data[v]), 4))

        else:
            plt_x.append(float(v))
            plt_y.append(round(float(data[v]), 4))

    return plt_x, plt_y
