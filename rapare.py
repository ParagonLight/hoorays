import numpy as np
import matrixFactorization as mf
import computeRecall as cr
import math
import random
import utils
import time
import computeMAE as mae
def predictionWithoutELO(kmeans, interviewSet, evaluationSet, itemsVector, usersVector, usersBias, itemsBias, mean, eloK, itemsSet, initModel):
    functions = {'kmeans':chooseInitVector,'average':chooseInitVectorFromInterviewList}
   
    RMSE = 0
    count = 0
    for userID in interviewSet:
        interviewList = interviewSet[userID]
        if initModel == 'kmeans':
            userVector = functions[initModel](kmeans, interviewList, itemsVector, itemsBias, usersBias, mean)
        else:# initModel = 'average':
            userVector = functions[initModel](interviewList, usersVector, itemsSet)

        #userVector = chooseInitVector(kmeans, interviewList, itemsVector, itemsBias, usersBias, mean)
#        userVector = chooseInitVectorFromInterviewList(interviewList, usersVector, itemsSet)
        evaluationList = evaluationSet[userID]
        for ratingInfo in evaluationList:
            rating = ratingInfo['rating']
            itemID = ratingInfo['itemID']
            itemBias = itemsBias[itemID - 1]
            itemVector = itemsVector[itemID - 1]
            predictedRating = mf.predictRating(userVector, itemVector, mean, 0, itemBias)
            if predictedRating > 5:
                predictedRating = 5
            elif predictedRating < 1:
                predictedRating = 1
            RMSE += mf.computeError(rating, predictedRating)**2
        count += len(evaluationList)

    RMSE /= count
    RMSE = math.sqrt(RMSE)
    print 'RMSE:', RMSE, 'count:', count



def train(mean, userNum, itemNum, featureK, trainSet, testSet, epochs, alpha,
        Lambdau, dataset, usersRatingCountedSet, itemsRatingCountedSet, init,
        usersVector, itemsVector, usersBias, itemsBias, Lambdauv, thetas, filePrefix, save, scale, Lambdav, hasLDA, tru,tri, teu, topN):
    if init:
        usersVector = mf.initFactorVectors(userNum, featureK)
        itemsVector = mf.initFactorVectors(itemNum, featureK)
        usersBias = mf.initBiases(userNum)
        itemsBias = mf.initBiases(itemNum)

    rmse, rmseNew, rmseOld, count, newCount, oldCount = mf.computeRMSE(testSet, usersVector, itemsVector, mean, usersBias, itemsBias, scale)
    print 'initialized RMSE:', rmse, 'count:', count
    maxRating = 5
    print maxRating
    lastRMSE = 1000
    random.shuffle(trainSet)
    print len(trainSet)
    trainSet = trainSet[:int(len(trainSet)*0.9)]
    print len(trainSet)
    for epoch in range(epochs):
        #if epoch % 50 == 0:
        if dataset == 'fm' or dataset =='delicious':
            if epoch > 499 and epoch % 500 == 1:
                print 'epoch:', epoch
                if hasLDA:
                    m = 'CTR'
                else:
                    m = 'MF'
                cr.recallWithParamsOnLearning(tru, tri, teu, topN, featureK, m, usersVector, itemsVector, thetas)
        random.shuffle(trainSet)
        start = time.time() 
        for row in trainSet:
            userID = row['userID']
            itemID = row['itemID']
            if row['rating'] == 0.01:
                rating = 0.0
            else:
                rating = float(row['rating'])
            userRatingSet = usersRatingCountedSet[userID]
            itemRatingSet = itemsRatingCountedSet[itemID]
            userVector = usersVector[int(userID) - 1]
            itemVector = itemsVector[int(itemID) - 1]
            userBias = usersBias[int(userID) - 1]
            itemBias = itemsBias[int(itemID) - 1]
            #theta = thetas[itemID-1]
            if len(thetas) != 0:
                theta = thetas[itemID-1]
            else:
                theta = [0.0] * len(userVector)
                theta = np.array(theta)
            predictedRating = mf.predictRating(userVector, itemVector, mean,
                    userBias, itemBias, scale)
            userVector, itemVector, userBias, itemBias = eloParamUpdating(userVector, itemVector, userBias, itemBias, Lambdau, alpha, predictedRating, rating, userRatingSet, itemRatingSet, maxRating, Lambdauv, theta, Lambdav)
            usersBias[int(userID) - 1] = userBias
            itemsBias[int(itemID) - 1] = itemBias
            usersVector[int(userID) - 1] = userVector
            itemsVector[int(itemID) - 1] = itemVector
        end = time.time()
        print end - start
        rmse, rmseNew, rmseOld, count, newCount, oldCount = mf.computeRMSE(testSet, usersVector, itemsVector, mean, usersBias, itemsBias, scale)
        rmseT, rmseNewT, rmseOldT, countT, newCountT, oldCountT = mf.computeRMSE(trainSet, usersVector, itemsVector, mean, usersBias, itemsBias, scale)
        print 'Iteration:',epoch,'train RMSE:', rmseT, 'test RMSE:', rmse, 'count:', count, 'MAE:', mae.computeMAE(testSet, usersVector, itemsVector)
        #if lastRMSE < rmse:
        #    lastRMSE = rmse
        #    break
        #else:
        #    print 'lastRMSE is',lastRMSE, ',continue optimization'
        #lastRMSE = rmse
    if save:
        folder = str(Lambdau)+'_'+str(Lambdav)+'_'+str(Lambdauv)
        import os
        savePre = filePrefix+folder+'/'
        if not hasLDA:
            savePre += 'noLDA_'
        if not os.path.exists(savePre):
            os.makedirs(savePre)
        print 'save data in', savePre
        utils.saveNumpyMatrix(savePre + 'RaP_usersVector.npy',usersVector)
        utils.saveNumpyMatrix(savePre + 'RaP_itemsVector.npy',itemsVector)
        utils.saveNumpyVector(savePre + 'RaP_usersBias.npy',usersBias)
        utils.saveNumpyVector(savePre + 'RaP_itemsBias.npy',itemsBias)

def eloParamUpdating(userVector, itemVector, userBias, itemBias,
        Lambdau, alpha, predictedRating, rating, userRatingSet, itemRatingSet, maxRating, Lambdauv, theta, Lambdav):
    userError = 0
    itemError = 0
    error = 0
    count = 0
    ratingSet = {1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
    rSet = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    #for r in range(1, maxRating + 1):
    for r in rSet:
        userRatingCount = userRatingSet[r] 
        itemRatingCount = itemRatingSet[r]
        Eij = sigmoidSij(predictedRating, r, None) 
        Sij = sigmoidSij(rating, r, None)
            #if SijModel == 'sigmoid': 
        #error += ratingCount
        base = 2 * (Sij - Eij) * exp(-(predictedRating
                - r))/(1+exp(-(predictedRating - r)))**2# * derivative
        error += (userRatingCount + itemRatingCount) * base
        #error += (userRatingCount) * base
        count += (userRatingCount + itemRatingCount)
    baseError = rating - predictedRating
    tempVector = userVector.copy()
    cij = 1.0
    if rating < 1:
        cij = 0.01
    baseError *= cij
    userVector = userVector + alpha * ((baseError + Lambdauv * error) * itemVector - Lambdau * userVector)

    itemVector = itemVector + alpha * ((baseError + Lambdauv * error) * tempVector - Lambdav * (itemVector - theta))
    #itemVector = itemVector + alpha * ((baseError + Lambdauv * error) * tempVector - Lambdav * itemVector)
    #userBias =  userBias + alpha * (baseError + Lambda1 * error - Lambda * userBias)
    #itemBias =  itemBias + alpha * (baseError + Lambda1 * error - Lambda * itemBias)
    return userVector, itemVector, 0, 0# userBias, itemBias


def prediction(kmeans, interviewSet, evaluationSet, itemsVector, usersVector,
        usersBias, itemsBias, mean, eloK, itemsSet, interviewKValue, SijModel,
        initModel, dataset, Lambda, usersCluster, countedRatingSet, featureK,
        usersVectorForItem):
    functions = {'kmeans':chooseInitVector,'average':chooseInitVectorFromInterviewList, 'mix':findVectorsInClosestCluster, 'random':randomVector}
    lastRMSE = 1000
    eloKInit = eloK

    usersInitVector = {}

    for userID in interviewSet:
        interviewList = interviewSet[userID]
        if initModel == 'kmeans':
            userVector, elector = functions[initModel](kmeans, interviewList, itemsVector, itemsBias, usersBias, mean)
        elif initModel == 'mix':
            userVector = functions[initModel](kmeans, interviewList,itemsSet, itemsVector, usersVector, itemsBias, usersBias, mean, usersCluster)
        elif initModel == 'average':
            userVector = functions[initModel](interviewList, usersVector,
                    itemsSet, usersVectorForItem)
        else:
            userVector = functions[initModel](featureK)
        usersInitVector[userID] = userVector

    maxRating = 5
    if dataset == 'eachMovie6':
        maxRating = 6
    print maxRating

    for index in range(100):
        RMSE = 0
        count = 0
        for userID in interviewSet:
            interviewList = interviewSet[userID]
            userVector = usersInitVector[userID]
            userVector = elo(userVector, interviewList, usersVector,
            itemsVector, usersBias, itemsBias, mean, eloKInit, itemsSet,
            interviewKValue, SijModel, dataset, Lambda, countedRatingSet)
            #eloKInit *= 0.9
            evaluationList = evaluationSet[userID]
            for ratingInfo in evaluationList:
                rating = ratingInfo['rating']
                itemID = ratingInfo['itemID']
                itemBias = itemsBias[itemID - 1]
                itemVector = itemsVector[itemID - 1]
                predictedRating = mf.predictRating(userVector, itemVector, mean, 0, itemBias)
                if predictedRating > maxRating:
                    predictedRating = maxRating
                elif predictedRating < 1:
                    predictedRating = 1
                RMSE += mf.computeError(rating, predictedRating)**2
            count += len(evaluationList)
            usersInitVector[userID] = userVector
        RMSE /= count
        RMSE = math.sqrt(RMSE)
        eloKInit *= 0.9
        #print 'RMSE: ',RMSE
        print 'Iteration:',index,'RMSE:', RMSE, 'count:', count
        if lastRMSE < RMSE or lastRMSE - RMSE < 0.00001:
            #continue
            break
        lastRMSE = RMSE
    print 'Last RMSE: ',lastRMSE
    return lastRMSE


def randomVector(featureK):
    return mf.initFactorVectors(1, featureK)

def chooseInitVectorFromInterviewList(interviewList, usersVector, itemsSet,
        usersVectorForItem):
    if len(interviewList) == 0:
        return np.zeros(usersVector[0].shape) 
    index = 0
    while len(interviewList) < index and interviewList[index]['itemID'] not in itemsSet:
        index += 1
    userVector = np.zeros(usersVector[interviewList[index]['userID'] - 1].shape)
    index += 1
    ratingCount = 0
    for ratingInfo in interviewList[index:]:
        count = 0
        rating = ratingInfo['rating']
        userID = ratingInfo['userID']
        itemID = ratingInfo['itemID']
        if int(rating) == 0:
            continue
        if itemID not in itemsSet:
            continue
        userItemVector = usersVectorForItem[itemID][int(rating)]

        '''
        userItemVector = np.zeros(usersVector[interviewList[0]['userID'] - 1].shape)
        otherRatings = itemsSet[itemID]
        for otherRating in otherRatings:
            ratingj = otherRating['rating']
            userjID = otherRating['userID']
            if ratingj == rating:
                userItemVector += usersVector[userjID - 1]
                count += 1
        if count > 0:
            userItemVector /= count
            ratingCount += 1
        '''
        userVector += userItemVector
        
    if ratingCount > 0:
        userVector /= ratingCount
    return userVector


def findVectorsInClosestCluster(kmeans, interviewList,
        itemsSet, itemsVector, usersVector, itemsBias, usersBias, mean, usersCluster):

    initUser, userKey = chooseInitVector(kmeans, interviewList, itemsVector, itemsBias, usersBias, mean)
    index = 0
    while len(interviewList) < index and interviewList[index]['itemID'] not in itemsSet:
        index += 1
    userVector = np.zeros(usersVector[interviewList[index]['userID'] - 1].shape)
    count = 0
    index += 1
    for ratingInfo in interviewList[index:]:
        rating = ratingInfo['rating']
        userID = ratingInfo['userID']
        itemID = ratingInfo['itemID']
        
        if itemID not in itemsSet:
            continue
        userItemVector = np.zeros(usersVector[interviewList[0]['userID'] - 1].shape)
        otherRatings = itemsSet[itemID]
        for otherRating in otherRatings:
            ratingj = otherRating['rating']
            userjID = otherRating['userID']
            userjVector = usersVector[userjID - 1]
            if ratingj == rating and usersCluster[userjID - 1] == userKey:    
                userItemVector += usersVector[userjID - 1]
                count += 1
        if count > 0:
            userItemVector /= count
        userVector += userItemVector
    userVector /= len(interviewList)
    return userVector

def chooseInitVector(kmeans, interviewList, itemsVector, itemsBias, usersBias, mean):
    minRMSE = 1000    
    elector = -1
    for key in kmeans:
        defaultVector = kmeans[key]
        RMSE = 0
        for ratingInfo in interviewList:
            rating = ratingInfo['rating']
            itemID = ratingInfo['itemID']
            itemBias = itemsBias[itemID - 1]
            itemVector = itemsVector[itemID - 1]
            predictedRating = mf.predictRating(defaultVector, itemVector, mean, 0, itemBias)
            
            RMSE += mf.computeError(rating, predictedRating)**2
        RMSE /= len(interviewList)
        RMSE = math.sqrt(RMSE)
        if RMSE < minRMSE:
            minRMSE = RMSE
            elector = key
#    print 'min rmse is:', minRMSE
    return kmeans[elector].copy(), elector


def computeExpectation(useriVector, itemVector, userBias, itemBias, mean, ratingj, predictedRatingi):
    if predictedRatingi == None:
        predictedRating = mf.predictRating(useriVector, itemVector, mean, userBias, itemBias)
    else:
        predictedRating = predictedRatingi
    #print predictedRating
    error = ratingj - predictedRating
#    print error
#    error /= 1
    if error < -300:
        Ei = 1
    elif error > 100:
        Ei = 0
    else:
        Ei = 1 / (1 + 10**error)
    return Ei



def elo(userVector, interviewList, usersVector, itemsVector, usersBias,
itemsBias, mean, eloK, itemsSet, interviewKValue, SijModel, dataset, Lambda,
countRatingSet):
    random.shuffle(interviewList)
    functions = {'linear':linearSij, 'oneZero':oneZeroSij, 'sigmoid':sigmoidSij}
#    error = 0
    for ratingInfo in interviewList[0:interviewKValue]:
        error = 0        
        rating = ratingInfo['rating']
        itemID = ratingInfo['itemID']
        itemBias = itemsBias[itemID - 1]
        itemVector = itemsVector[itemID - 1]
        if itemID not in itemsSet or itemID not in countRatingSet:
            continue
        otherRatings = itemsSet[itemID]
        predictedRatingi = mf.predictRating(userVector, itemVector, mean, 0, itemBias)
        
#        defaultRating = mf.predictRating(userVector, itemVector, mean, 0, itemBias)
        '''
        Eij = computeExpectation(userVector, itemVector, 0, itemBias, mean, rating)
        Sij = 1
        if defaultRating < rating:
            Sij = 0
        error += Sij - Eij

        '''
#        print 'lens:', len(otherRatings)
        count = 0
        '''
        ratingSet = {1:0,2:0,3:0,4:0,5:0,6:0}
        for otherRating in otherRatings:
            ratingj = int(otherRating['rating'])
            ratingSet[ratingj] += 1
        '''
        ratingSet = countRatingSet[itemID]
        for ratingj in ratingSet.keys():
            ratingCount = ratingSet[ratingj]
            Eij = computeExpectation(userVector, itemVector, 0, itemBias, mean, ratingj, predictedRatingi)
            Sij = functions[SijModel](rating, ratingj, dataset)
            if SijModel == 'sigmoid': 
                error += ratingCount * 2 * (Sij - Eij) * 10**((predictedRatingi - ratingj))*math.log(10)/(1+10**(predictedRatingi - ratingj))**2# * derivative
                count += ratingCount
            elif SijModel == 'oneZero':
                error += ratingCount * 2 * (Sij - Eij)
                count += ratingCount
        '''
        for otherRating in otherRatings:
            if count > 200:
                break
            ratingj = otherRating['rating']
            userjID = otherRating['userID']
            userjVector = usersVector[userjID - 1]
#            userjBias = usersBias[userID]        
#            predictedRatingj = mf.predictRating(userjVector, itemVector, mean, 0, itemBias)
#            Eij = computeExpectation(userVector, itemVector, 0, itemBias, mean, ratingj)
            Eij = computeExpectation(userVector, itemVector, 0, itemBias, mean, ratingj, predictedRatingi)
#            derivative = computeDerivativeOfExpectationFunction(predictedRatingi, predictedRatingj)
#            Sij = oneZeroSij(rating, ratingj)
#            Sij = linearSij(rating, ratingj)
#            Sij = sigmoidSij(ratingj - rating)
            Sij = functions[SijModel](rating, ratingj, dataset)
#            print Eij, Sij, predictedRatingj, ratingj, rating, predictedRatingi
            #error += Sij - Eij# * derivative
            error += 2*(Sij - Eij) * 10**((predictedRatingi -
                ratingj))*math.log(10)/(1+10**(predictedRatingi - ratingj))**2# * derivative
            #error += 2*(Sij - Eij) * 10**((predictedRatingi -
            #    ratingj))*math.log(10)/(1+10**(predictedRatingi - ratingj))**2# * derivative
            count += 1
#        userVector = userVector + eloK * error * itemVector / np.dot(itemVector,itemVector)
        '''
        userVector = userVector + eloK * (error * itemVector - Lambda *
                  userVector)
        #eloK *= 0.9
    return userVector



def computeDerivativeOfExpectationFunction(ratingi, ratingj):
    x = ratingj - ratingi
    denominator = (1 + 10**x)**2
    numerator = -(10**x) * math.log(10)
    return numerator / denominator
    


def oneZeroSij(ratingi, ratingj, dataset):
    Sij = 1
    if ratingi < ratingj:
        Sij = 0
    elif ratingi - ratingj == 0:
        Sij = 0.5
    return Sij

def linearSij(ratingi, ratingj, dataset):
    gap = 4.0
    if dataset == 'eachMovie6':
        gap = 5.0
    
    error = ratingi - ratingj
    return (error + gap) / (2 * gap)

def exp(x):
    if x < -300:
        result = 0
    elif x > 100:
        result = 1
    else:
        result = math.exp(x)
    return result

def sigmoidSij(ratingi, ratingj, dataset):
    error = ratingj - ratingi
    result = 1/(1 + exp(error))
    return result
