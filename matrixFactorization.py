import random
import numpy as np
import math
import time
import utils
import dataAnalysis as da
import computeMAE as mae


def predictRating(pu, qi, mean, userBias, itemBias, data):
    predictedRating = mean + np.dot(pu,qi) + userBias + itemBias
    if data == 1:
        maxR = 1
        minR = 0
    else:
        maxR = 5
        minR = 1
    if predictedRating > maxR:
        predictedRating = maxR
    elif predictedRating < minR:
        predictedRating = minR
    return predictedRating


def computeError(realRating, predictedRating):
    return realRating - predictedRating

def computeRMSE(dataset, userVectors, itemVectors, mean, usersBias, itemsBias, data):
    rmse = 0
    rmseOld = 0
    oldCount = 0
    rmseNew = 0
    newCount = 0
    count = 0
    for row in dataset:
        userID = int(row['userID'])
        itemID = int(row['itemID'])
        rating = float(row['rating'])
        # Since list index in python starts from 0 and user ID 
        #in MovieLens dataset start from 1, all IDs should minus one.
        userVector = userVectors[userID - 1]
        itemVector = itemVectors[itemID - 1]
        userBias = usersBias[userID - 1]
        itemBias = itemsBias[itemID - 1]
        predictedRating = predictRating(userVector, itemVector, mean, userBias, itemBias, data)
        error = computeError(rating, predictedRating)
        temp = error**2
        rmse += temp
#        if setType == 'test':
#            if userID not in userIDDict or itemID not in itemIDDict:
#                rmseNew += temp
#                newCount += 1
#            else:
#                rmseOld += temp
#                oldCount += 1
        count += 1
    rmse /= count#len(dataset)
    rmse = math.sqrt(rmse)
#    if setType == 'test':
#        rmseNew /= newCount
#        rmseNew = math.sqrt(rmseNew)
#        rmseOld /= oldCount 
#        rmseOld = math.sqrt(rmseOld)

    return rmse, rmseNew, rmseOld, count, newCount, oldCount

def initFactorVectors(rows, columns):
    return np.random.randn(rows, columns) / math.sqrt(columns)


def updateParams(pu, qi, LambdaU, LambdaV, alpha, error, userBias, itemBias, theta):
#    tempu = np.copy(pu)
    #userBias += alpha * (error - Lambda * userBias)
    #itemBias += alpha * (error - Lambda * itemBias)
    temp = pu.copy()
    pu += alpha * (error * qi - LambdaU * pu)    
    if theta != None:
        qi += alpha * (error * temp - LambdaV * (qi-theta))
    else:
        qi += alpha * (error * temp - LambdaV * qi)
    return 0,0# itemBias
    #return userBias, itemBias
#    pu += deltaPu
#    qi += deltaQi
#    deltaPu = alpha * np.add(error * qi, -Lambda * pu)
#    print 'deltaPu:', deltaPu.shape
#    deltaQi = alpha * np.add(error * pu, -Lambda * qi)
#    print 'deltaQi:', deltaQi.shape
#    print pu
#    pu = np.add(pu, deltaPu)
#    print pu
#    qi = np.add(qi, deltaQi)


def initBiases(number):
    return [0] * number

def train(userNum, itemNum, featureK, trainSet, testSet, epochs, LambdaU, LambdaV, alpha, mean, filePrefix, isSave, thetas, data):
    ratio = 0.75
    userVectors = initFactorVectors(userNum, featureK)
    itemVectors = initFactorVectors(itemNum, featureK)
    usersBias = initBiases(userNum)
    itemsBias = initBiases(itemNum)
    

    finalEpoch = epochs

    lastRMSE = 1000
    totalStart = time.time()
    for epoch in range(finalEpoch):
        random.shuffle(trainSet)    
        start = time.time()
        for row in trainSet:
            userID = int(row['userID'])
            itemID = int(row['itemID'])
            rating = float(row['rating'])
            if len(thetas) != 0:
                theta = thetas[itemID-1]
            else:
                theta = None
            # Since list index in python starts from 0 and user ID 
            #in MovieLens dataset start from 1, all IDs should minus one.
            userVector = userVectors[userID - 1]
            itemVector = itemVectors[itemID - 1]
            userBias = usersBias[userID - 1]
            itemBias = itemsBias[itemID - 1]
            predictedRating = predictRating(userVector, itemVector, mean, userBias, itemBias, data)
            error = computeError(rating, predictedRating)
            usersBias[userID - 1], itemsBias[itemID - 1] = updateParams(userVector, itemVector, LambdaU, LambdaV, alpha, error, userBias, itemBias, theta)
        trainRMSE, rmseNew, rmseOld, count, newCount, oldCount = computeRMSE(trainSet, userVectors, itemVectors, mean,  usersBias, itemsBias, data)
        testRMSE, testRmseNew, testRmseOld, count, newCount, oldCount = computeRMSE(testSet, userVectors, itemVectors, mean,  usersBias, itemsBias, data)
        end = time.time()
        print epoch, trainRMSE, testRMSE, end -start, mae.computeMAE(testSet, userVectors, itemVectors)
        #print 'Epoch',epoch, 'finished', 'RMSE in training set:', trainRMSE, 'RMSE in testing set:', testRMSE, count, 'RMSE with old item and user in testing set:',testRmseOld, oldCount, 'RMSE with new item or new user in testing set:', testRmseNew, newCount, 'time cost:', end - start
        #if lastRMSE < testRMSE or lastRMSE - testRMSE < 0.00001:
        #    break
        if lastRMSE > testRMSE:
            lastRMSE = testRMSE
        #else:
        #    break
    totalEnd = time.time()
    print 'RMSE of MF for testing set:', testRMSE 
    print 'time cost: ', totalEnd - totalStart
        #trainRMSE, rmseNew, rmseOld, count, newCount, oldCount = computeRMSE(trainSet, userVectors, itemVectors, trainUserIDDict, trianItemIDDict, 'train', mean,  usersBias, itemsBias)
        #testRMSE, testRmseNew, testRmseOld, count, newCount, oldCount = computeRMSE(testSet, userVectors, itemVectors, trainUserIDDict, trianItemIDDict, 'test', mean,  usersBias, itemsBias)
        #print 'Epoch',epoch, 'finished', 'RMSE in training set:', trainRMSE, 'RMSE in testing set:', testRMSE, count, 'RMSE with old item and user in testing set:',testRmseOld, oldCount, 'RMSE with new item or new user in testing set:', testRmseNew, newCount, 'time cost:', end - start
    if isSave:
        folder = str(LambdaU)+'_'+str(LambdaV)
        import os
        savePre = filePrefix+folder+'/'
        print 'save data in', savePre
        if len(thetas) != 0:
            savePre += 'CTR_'
        else:
            savePre += 'MF_'
        if not os.path.exists(savePre):
            os.makedirs(savePre)
        print 'save params...'
        utils.saveNumpyMatrix(savePre + 'usersVector.npy',userVectors)
        utils.saveNumpyMatrix(savePre + 'itemsVector.npy',itemVectors)
        utils.saveNumpyVector(savePre + 'usersBias.npy',usersBias)
        utils.saveNumpyVector(savePre + 'itemsBias.npy',itemsBias)
        print 'params saved'

    return testRMSE 
