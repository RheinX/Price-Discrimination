import numpy as np


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

    right_rate=0.0
    size = len(error)
    result = {}
    for i in range(size):
        v = float(error[i]) / float(label[i])

        offset = str(round(v, 2))

        if offset not in result:
            result[offset] = 0

        result[offset] += 1

    for v in result:
        num=result[v]
        result[v] = float(result[v]) / size

        if float(v)<=rate:
            right_rate+=num

    return result, float(right_rate)/size


def plt_array(data, rate, filter=True):
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
        if filter :
            if float(v) <= rate:
                plt_x.append(float(v))
                plt_y.append(round(float(data[v]), 4))

        else:
            plt_x.append(float(v))
            plt_y.append(round(float(data[v]), 4))

    return plt_x, plt_y
