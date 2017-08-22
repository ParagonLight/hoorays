import matrixFactorization as mf
import utils
import math
import ctrUtil as cu
import numpy as np
import operator
import sys
def rocAll(trainUser, trainItem, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, featureK, model, thetas):
    se = [0.0 for i in range(11)]
    sp = [0.0 for i in range(11)]
    noUse = 0
    count = 0
    itemNotInTrain = {}
    print len(testUser.keys())
    for item in trainItem.keys():
        if len(trainItem[item]) == 0:
            itemNotInTrain[item] = 1
    for u in testUser.keys():
        if count % 100 == 0:
            print count
        #    print se, sp 
        count += 1
        userSe, userSp = roc(u, trainUser, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, itemNotInTrain, thetas, model, featureK)
        if userSe[0] == -1:
            noUse += 1
        else:
            se = [se[i] + userSe[i] for i in range(len(se))]
            sp = [sp[i] + userSp[i] for i in range(len(sp))]
            #se += userSe#[se[i] + userSe[i] for i in range(len(tops))]
            #sp += userSp#[sp[i] + userSp[i] for i in range(len(tops))]
    #tops = [float(n)/(len(testUser.keys())-noUse) for n in tops]
    sp = [float(n)/(len(testUser.keys())-noUse) for n in sp]
    se = [float(n)/(len(testUser.keys())-noUse) for n in se]
    print se
    print sp
    #print topN
    

def roc(userID, trainUser, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, itemNotInTrain, thetas, model, featureK):
    train = trainUser[userID]
    test = testUser[userID]
    tops = [0 for i in topN]
    if len(test) == 0:
        return [-1 for i in topN], [-1]
    userVec = usersVector[userID - 1]
    #userBias = usersBias[userID - 1]
    likes = {}
    
    itemInTrain = {}
    for r in train:
        itemInTrain[r['itemID']] = 1
    itemInTest = {}
    for r in test:
        itemInTest[r['itemID']] = 1



    for i in range(1,itemNum+1):
        if i in itemInTrain:
            continue
        if i not in itemNotInTrain:
            itemVec = itemsVector[i-1]
        else:
            if model == 'MF':
                #itemVec = [0.0] * featureK
                itemVec = np.random.rand(featureK)/math.sqrt(featureK)
                itemVec = np.array(itemVec)
            else:
                itemVec  = thetas[i-1]
        #itemBias = itemsBias[i-1]
        #predictedRating = np.dot(userVec,itemVec) + userBias + itemBias
        predictedRating = np.dot(userVec,itemVec)
        likes[i] = predictedRating
    #print len(likes)
    '''
    N = topN[len(topN)-1]
    sortedLikes = []
    for i in range(N):
        tempId = -1
        tempValue = -1.0
        for key in likes.keys():
            v = likes[key]
            if v > tempValue:
                tempValue = v
                tempId = key
        sortedLikes.append(key)
        likes[key] = -1.0
    '''
    '''
    sortedLikes = sorted(likes.items(), key=operator.itemgetter(1))
    sortedLikes.reverse()
    #print len(sortedLikes), len(itemInTest.keys())
    for i in range(len(topN)):
        n = topN[i]
        subLikes = sortedLikes[:n]
        #print subLikes
        #print itemInTest
        for s in subLikes:
            if s[0] in itemInTest.keys():#for sorted function
            #if s in itemInTest.keys():
                tops[i] += 1
    tops = [float(t)/len(itemInTest) for t in tops]
    '''
    cuts = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    cuts.reverse()
    sensitivity = []
    specificity = []

    for cut in cuts:
        se = 0.0
        sp = 0.0
        for i in likes.keys():
            l = 0.0
            if likes[i] < 0:
                l = 0.0
            elif likes[i] > 1:
                l = 1.0
            else:
                l = likes[i]
            if i in itemInTest and l >= cut:
                se += 1
            elif i not in itemInTest and l >= cut:
                sp += 1
        sensitivity.append(se/len(itemInTest))
        specificity.append(sp/(len(likes)-len(itemInTest)))
    

    #return tops
    return sensitivity, specificity

def recallAll(trainUser, trainItem, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, featureK, model, thetas):
    tops = [0.0 for i in topN]
    noUse = 0
    count = 0
    itemNotInTrain = {}
    print len(testUser.keys())
    for item in trainItem.keys():
        if len(trainItem[item]) == 0:
            itemNotInTrain[item] = 1
    for u in testUser.keys():
        print count
        if count % 100 == 0:
            print count
        count += 1
        topNuser = recall(u, trainUser, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, itemNotInTrain, thetas, model, featureK)
        if topNuser[0] == -1:
            noUse += 1
        else:
            tops = [tops[i] + topNuser[i] for i in range(len(tops))]
    tops = [float(n)/(len(testUser.keys())-noUse) for n in tops]
    #print topN
    print tops
    

def recall(userID, trainUser, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, itemNotInTrain, thetas, model, featureK):
    train = trainUser[userID]
    test = testUser[userID]
    tops = [0 for i in topN]
    if len(test) == 0:
        return [-1 for i in topN]
    userVec = usersVector[userID - 1]
    #userBias = usersBias[userID - 1]
    likes = {}
    
    itemInTrain = {}
    for r in train:
        itemInTrain[r['itemID']] = 1
    itemInTest = {}
    for r in test:
        itemInTest[r['itemID']] = 1



    for i in range(1,itemNum+1):
        if i in itemInTrain:
            continue
        if i not in itemNotInTrain:
            itemVec = itemsVector[i-1]
        else:
            if model == 'MF':
                #itemVec = [0.0] * featureK
                itemVec = np.random.rand(featureK)/math.sqrt(featureK)
                itemVec = np.array(itemVec)
            else:
                itemVec  = thetas[i-1]
        #itemBias = itemsBias[i-1]
        #predictedRating = np.dot(userVec,itemVec) + userBias + itemBias
        predictedRating = np.dot(userVec,itemVec)
        likes[i] = predictedRating
    #print len(likes)
    sortedLikes = sorted(likes.items(), key=operator.itemgetter(1))
    sortedLikes.reverse()
    for i in range(len(topN)):
        n = topN[i]
        subLikes = sortedLikes[:n]
        #print subLikes
        #print itemInTest
        for s in subLikes:
            if s[0] in itemInTest.keys():
                tops[i] += 1
    tops = [float(t)/len(itemInTest) for t in tops]
    return tops



def recallWithParams(filePrefix, rootPath, tru, tri, teu, topN, featureK, model):
    usersVectorFile = filePrefix + 'usersVector.npy'
    usersBiasFile = filePrefix + 'usersBias.npy'
    itemsVectorFile = filePrefix + 'itemsVector.npy'
    itemsBiasFile = filePrefix + 'itemsBias.npy'
    thetaPath = rootPath + 'final.gamma' + str(featureK)
    usersVector = utils.loadNumpyMatrix(usersVectorFile)
    itemsVector = utils.loadNumpyMatrix(itemsVectorFile)
    usersBias = None#utils.loadNumpyMatrix(usersBiasFile)
    itemsBias = None#utils.loadNumpyMatrix(itemsBiasFile)

    trainUser, trainData = cu.loadData(tru, 'user')
    trainItem, noUse= cu.loadData(tri, 'item')
    theta = cu.loadTheta(thetaPath)
    userNum = len(trainUser.keys())
    itemNum = len(trainItem.keys())
    noUse = None

    #print 'load test data...'
    testUser, testData = cu.loadData(teu, 'user')
    #testItem, testDataNoUse = cu.loadData(tei, 'item')
    noUse = None
    
    #recallAll(trainUser, trainItem, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, featureK, model, theta)
    rocAll(trainUser, trainItem, testUser, itemNum, topN, usersVector, itemsVector, usersBias, itemsBias, featureK, model, theta)

def recallModel(rootPath, featureK, tru, tri, teu, topN):
    if MF:
        filePrefix = rootPath + str(featureK) + 'MF_'
        model = 'MF'
    elif CTR:
        model = 'CTR'
        filePrefix = rootPath + str(featureK) + 'CTR_'
    else:#RAPARE
        model = 'RAPARE'
        filePrefix = rootPath + str(featureK) + 'RaP_'

    recallWithParams(filePrefix, rootPath, tru, tri, teu, topN, featureK, model)

def recallParamSelection(rootPath, featureK, tru, tri, teu, topN, param, hasLDA):
    if MF:
        model = 'MF'
    elif CTR:
        model = 'CTR'
    filePrefix = rootPath + str(featureK) + param + model + '_' 
    print filePrefix
    recallWithParams(filePrefix, rootPath, tru, tri, teu, topN, featureK, model)

if __name__ == '__main__':

    argv = sys.argv[1:]
    featureK = 200 
    data = argv[0] 
    #rootPath = 'ctrdata/' + data + rate + '/'
    rootPath = 'ctrdata/' + data +'/'
    MF = True 
    CTR = False 
    RAPARE = False 
    hasLDA = False
    topN = [10,15,20,25,30,35,40,45,50]
    #topN = [n for n in range(topN[0], topN[1],5)]
    print topN
    tenPlus = False 
    if tenPlus:
        p = '-10plus'
    else:
        p = ''
    tru = rootPath + data + '-Train-User'+p+'.dat'
    tri = rootPath + data + '-Train-Item'+p+'.dat'
    teu = rootPath + data + '-Test-User'+p+'.dat'
    Lambdaus = [0.1]#,0.05,0.1,0.001]
    Lambdavs = [0.1]#,0.05,0.1,0.001]
    params = [str(u)+'_'+str(v)+'/' for v in Lambdavs for u in Lambdaus]
    print params
    for param in params:
        print param
        recallParamSelection(rootPath, featureK, tru, tri, teu, topN, param, hasLDA)
