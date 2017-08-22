def loadData(filePath, dataType):
    f = open(filePath, 'r')
    dataSet = {}
    dataList = []
    N = 1
    type1 = 'userID'
    type2 = 'itemID'
    if dataType == 'item':
        type1, type2 = type2, type1

    total = 0.0
    count = 0
    for line in f:
        line = line.strip()
        strs = line.split(' ')
        num = int(strs[0])
        strs = strs[1:]
        dataSet[N] = []
        for ele in strs:
            es = ele.split(':')
            ID = int(es[0])
            rating = float(es[1])
            total += rating
            ratingInfo = {}
            ratingInfo[type1] = N
            ratingInfo[type2] = (ID + 1)
            ratingInfo['rating'] = rating
            dataList.append(ratingInfo)
            dataSet[N].append(ratingInfo)
        count += num
        N += 1
    print total / count 
    return dataSet, dataList


def loadTheta(filePath):
    import numpy as np
    f = open(filePath, 'r')
    theta = []
    N = 1
    for line in f:
        line = line.strip()
        strs = line.split(' ')
        floats = [float(s) for s in strs]
        total = sum(floats)
        floats = [s/total for s in floats]
        theta.append(floats)

    theta = np.array(theta)
    return theta

