def error_rate(predict, label):
    error=abs(predict-label)

    size = len(error)
    result={}
    for i in range(size):
        v=float(error[i])/float(label[i])

        offset=str(round(v,2))

        if offset not in result:
            result[offset]=0

        result[offset]+=1


    for v in result:
        result[v]=float(result[v])/size

    return result