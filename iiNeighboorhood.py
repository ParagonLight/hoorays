import time
import matrixFactorization as mf
import math
import utils
import operator


def computeMean(userIDDict, itemIDDict):
    itemMeanDict = generateItemsMeans(itemIDDict)
    userMeanDict = generateUsersMeans(userIDDict)
    return itemMeanDict, userMeanDict

def computeuu(userIDDict, itemIDDict, userMeanDict):
#    itemMeanDict = generateItemsMeans(itemIDDict)
    userIDVector, userIDRlengths = generateUsersVectors(userIDDict, userMeanDict)
    usersSimilarities = computeSimilarities(userIDVector, userIDRlengths)
    return usersSimilarities

def computeuuWithFile(userIDDict, itemIDDict, userMeanDict, filePrefix):
    userIDVector, userIDRLengths = generateUsersVectors(userIDDict, userMeanDict)
    computeSimilaritiesWithFileSaving(userIDVector, userIDRLengths,
            filePrefix)
    
def generateItemsMeans(userIDDict):
    userMeanDict = {}
    for userID in userIDDict.keys():
        userRatings = userIDDict[userID]
        total = 0.0
        for ratingInfo in userRatings:
            total += ratingInfo['rating']
        mean = total / len(userRatings)
        userMeanDict[userID] = mean
    return userMeanDict

def generateUsersVectors(userIDDict, userMeanDict):
    userIDVector = {}
    userIDRLengths = {}
    for userID in userIDDict.keys():
        userList = userIDDict[userID]
        userDict = {}
        userRLength = 0.0
        userMean = userMeanDict[userID]        
        for ratingInfo in userList:
            itemID = ratingInfo['itemID']
            rating = ratingInfo['rating']
            userRLength += (rating - userMean)**2
            userDict[itemID] = rating - userMean
        userRLength = math.sqrt(userRLength)
        if userRLength == 0:
            userRLength = 0.000001
        assert userRLength > 0
        userIDRLengths[userID] = userRLength
        userIDVector[userID] = userDict
    return userIDVector, userIDRLengths


def predictionuuWithFile(trainItemIDDict, testItemIDDict,
        userMeanDict, filePrefix):
#    threshold = 0.001
#    K = 20
    RMSE = 0
    count = 0

    for itemID in testItemIDDict.keys():
        predictionList = testItemIDDict[itemID]
        if itemID not in trainItemIDDict:
            continue
        trainList = trainItemIDDict[itemID]
        for ratingInfo in predictionList:
            userIDi = ratingInfo['userID']
            ratingi = ratingInfo['rating']
            if userIDi not in userSimilarities:
                continue
            useriMean = userMeanDict[userIDi]        
            
            useriSimilarityList = utils.loadDictFromCSVFile(filePrefix+str(userIDi))
            denominator = 0.0
            numerator = 0.0
            for trainRatingInfo in trainList:
                userIDj = trainRatingInfo['userID']
                ratingj = trainRatingInfo['rating']
                userjMean = userMeanDict[userIDj]
                similarityij = useriSimilarityList[userIDj]
                denominator += math.fabs(similarityij)
                numerator += (ratingj - userjMean) * similarityij
            predictedRating = useriMean + numerator / denominator
            RMSE += (predictedRating - ratingi)**2
            #print count
            count += 1
#    print count
    RMSE /= count
    RMSE = math.sqrt(RMSE)

def updateSimilarityItem(interviewList, dataset, alpha, Lambda, similarities, means, trainList):
    for ratingInfo in interviewList:
        ratingi = ratingInfo['rating']
        itemID = ratingInfo['itemID']
        userID = ratingInfo['userID']
        predictedRatingi = predictRatingItem(means, itemID, similarities, trainList)
        tSimi = 0.0
        tSimiR = 0.0
        for ratingjInfo in trainList:
            itemjID = ratingjInfo['itemID']
            ratingj = ratingjInfo['rating']
            if itemID != itemjID:
                simiIJ = similarities[itemID][itemjID]
                itemJmean = means[itemjID]
                tSimi += math.fabs(simiIJ)
                tSimiR += simiIJ * (ratingj - itemJmean)

        if len(trainList) == 0:
            tSimi = 1.0
            tSimiR = 0
        if tSimi < 1e-7:
            tSimi = 1.0
            tSimiR = 0
        for ratingjInfo in trainList:
            ratingj = ratingjInfo['rating']
            itemjID = ratingjInfo['itemID']
            if itemID != itemjID:
                itemJmean = means[itemjID]
                simi = similarities[itemID][itemjID]
                Eij = sigmoid(predictedRatingi, ratingj)
                Sij = sigmoid(ratingi, ratingj)
                simi -= alpha * ((Sij - Eij) * (10**(ratingj - predictedRatingi)) / (1 + 10**(ratingj - predictedRatingi))**2 * ((ratingj - itemJmean) * tSimi - tSimiR) / ((tSimi)**2) - Lambda * simi) 
                similarities[itemID][itemjID] = simi
                similarities[itemjID][itemID] = simi
    return similarities


def predictRatingItem(means, itemID, similarities, trainList):
    itemiMean = means[itemID]        
    itemiSimilarityList = similarities[itemID]
    denominator = 0.0
    numerator = 0.0
    for trainRatingInfo in trainList:
        itemIDj = trainRatingInfo['itemID']
        ratingj = trainRatingInfo['rating']
        if itemID != itemIDj:
            itemjMean = means[itemIDj]
            similarityij = itemiSimilarityList[itemIDj]
            denominator += math.fabs(similarityij)
            numerator += (ratingj - itemjMean) * similarityij
    if denominator == 0:
        predictedRating = itemiMean
    else:
        predictedRating = itemiMean + numerator / denominator
    return predictedRating


def updateSimilaritiesItem(similarities, means, userIDDict, trainItemIDDict, alpha, Lambda, dataset, interviewSet, evaluationSet):
    maxRating = 5
    if dataset == 'eachMovie6':
        maxRating = 6
    print maxRating
    totalTime = 0
    lastRMSE = 1000
    for index in range(100):
        RMSE = 0
        count = 0
        jump = 0
        for userID in interviewSet:
            start = time.time() 
            interviewList = interviewSet[userID]
            if int(userID) in userIDDict:
                similarities = updateSimilarityItem(interviewList, dataset, alpha, Lambda, similarities, means, userIDDict[int(userID)])
            #eloKInit *= 0.9
            end = time.time()
            totalTime += (end - start)
        for userID in interviewSet:
            evaluationList = evaluationSet[userID]
            if int(userID) in userIDDict:
                trainList = userIDDict[int(userID)]
                for ratingInfo in evaluationList:
                    rating = ratingInfo['rating']
                    itemID = ratingInfo['itemID']
                    if itemID in means:
                        predictedRating = predictRatingItem(means, itemID, similarities, trainList)
                        if predictedRating > maxRating:
                            predictedRating = maxRating
                        elif predictedRating < 1:
                            predictedRating = 1
                        RMSE += mf.computeError(rating, predictedRating)**2
                    else:
                        jump += 1
                count += len(evaluationList)
        RMSE /= (count - jump)
        RMSE = math.sqrt(RMSE)
        #eloKInit *= 0.9
        #print 'RMSE: ',RMSE
        print 'Iteration:',index,'RMSE:', RMSE, 'count:', count
        if lastRMSE < RMSE or lastRMSE - RMSE < 0.000001:
            #continue
            break
        lastRMSE = RMSE
    print totalTime
    print 'Last RMSE: ',lastRMSE
    return lastRMSE, totalTime

def updateSimilarity(interviewList, dataset, alpha, Lambda, similarities, means, trainDict):
    for ratingInfo in interviewList:
        ratingi = ratingInfo['rating']
        itemID = ratingInfo['itemID']
        userID = ratingInfo['userID']
        if itemID not in trainDict:
            continue
        predictedRatingi = predictRating(means, userID, similarities, trainDict[itemID])
        tSimi = 0.0
        tSimiR = 0.0
        for ratingjInfo in trainDict[itemID]:
            uservID = ratingjInfo['userID']
            ratingj = ratingjInfo['rating']
            if userID == uservID:
                continue
            simiUV = similarities[userID][uservID]
            userVmean = means[uservID]
            tSimi += math.fabs(simiUV)
            tSimiR += simiUV * (ratingj - userVmean)

        if len(trainDict[itemID]) == 0:
            tSimi = 1.0
            tSimiR = 0
        if tSimi < 1e-7:
            tSimi = 1.0
            tSimiR = 0
        for ratingjInfo in trainDict[itemID]:
            ratingj = ratingjInfo['rating']
            uservID = ratingjInfo['userID']
            if userID == uservID:
                continue
            userVmean = means[uservID]
            simi = similarities[userID][uservID]
            Eij = sigmoid(predictedRatingi, ratingj)
            Sij = sigmoid(ratingi, ratingj)
            simi += alpha * ((Sij - Eij) * (10**(ratingj - predictedRatingi)) / (1 + 10**(ratingj - predictedRatingi))**2 * ((ratingj - userVmean) * tSimi - tSimiR) / ((tSimi)**2) - Lambda * simi) 
            similarities[userID][uservID] = simi
            similarities[uservID][userID] = simi
    return similarities


def sigmoid(ratingi, ratingj):
    temp = ratingi - ratingj
    result = 1/(1 + math.exp(-temp))
    return result

def predictRating(means, userID, similarities, trainList):
    useriMean = means[userID]        
    useriSimilarityList = similarities[userID]
    denominator = 0.0
    numerator = 0.0
    for trainRatingInfo in trainList:
        userIDj = trainRatingInfo['userID']
        ratingj = trainRatingInfo['rating']
        if userID == userIDj:
            continue
        userjMean = means[userIDj]
        similarityij = useriSimilarityList[userIDj]
        denominator += math.fabs(similarityij)
        numerator += (ratingj - userjMean) * similarityij
    if denominator == 0:
        predictedRating = useriMean
    else:
        predictedRating = useriMean + numerator / denominator
    return predictedRating


def updateSimilarities(similarities, means, trainUserIDDict, trainItemIDDict, alpha, Lambda, dataset, interviewSet, evaluationSet):
    maxRating = 5
    if dataset == 'eachMovie6':
        maxRating = 6
    print maxRating
    totalTime = 0
    lastRMSE = 1000
    for index in range(100):
        RMSE = 0
        count = 0
        jump = 0
        for userID in interviewSet:
            start = time.time() 
            interviewList = interviewSet[userID]
            similarities = updateSimilarity(interviewList, dataset, alpha, Lambda, similarities, means, trainItemIDDict)
            #eloKInit *= 0.9
            end = time.time()
            totalTime += (end - start)
        for userID in interviewSet:
            evaluationList = evaluationSet[userID]
            for ratingInfo in evaluationList:
                rating = ratingInfo['rating']
                itemID = ratingInfo['itemID']
                userID = ratingInfo['userID']
                if itemID not in trainItemIDDict:
                    jump += len(evaluationList)
                    continue
                trainList = trainItemIDDict[itemID]
                predictedRating = predictRating(means, userID, similarities, trainList)
                if predictedRating > maxRating:
                    predictedRating = maxRating
                elif predictedRating < 1:
                    predictedRating = 1
                RMSE += mf.computeError(rating, predictedRating)**2
            count += len(evaluationList)
        RMSE /= (count - jump)
        RMSE = math.sqrt(RMSE)
        #eloKInit *= 0.9
        #print 'RMSE: ',RMSE
        print 'Iteration:',index,'RMSE:', RMSE, 'count:', count
        if lastRMSE < RMSE or lastRMSE - RMSE < 0.0001:
            #continue
            break
        lastRMSE = RMSE
    print totalTime
    print 'Last RMSE: ',lastRMSE
    return lastRMSE, totalTime

def predictionuu(userSimilarities, trainItemIDDict, testItemIDDict, userMeanDict):
#    threshold = 0.001
#    K = 20
    RMSE = 0
    count = 0

    for itemID in testItemIDDict.keys():
        predictionList = testItemIDDict[itemID]
        if itemID not in trainItemIDDict:
            continue
        trainList = trainItemIDDict[itemID]
        for ratingInfo in predictionList:
            userIDi = ratingInfo['userID']
            ratingi = ratingInfo['rating']
            if userIDi not in userSimilarities:
                continue
            useriMean = userMeanDict[userIDi]        
            useriSimilarityList = userSimilarities[userIDi]
            denominator = 0.0
            numerator = 0.0
            for trainRatingInfo in trainList:
                userIDj = trainRatingInfo['userID']
                ratingj = trainRatingInfo['rating']
                userjMean = userMeanDict[userIDj]
                similarityij = useriSimilarityList[userIDj]
                denominator += math.fabs(similarityij)
                numerator += (ratingj - userjMean) * similarityij
            if denominator == 0:
                predictedRating = useriMean
            else:
                predictedRating = useriMean + numerator / denominator
            RMSE += (predictedRating - ratingi)**2
            #print count
            count += 1
#    print count
    RMSE /= count
    RMSE = math.sqrt(RMSE)
    print 'RMSE: ', RMSE, 'count: ', count
    return RMSE


def computeii(userIDDict, itemIDDict, itemMeanDict):
#    userMeanDict = generateUsersMeans(userIDDict)
    itemIDVector, itemIDRlengths = generateItemsVectors(itemIDDict, itemMeanDict)
    itemsSimilarities = computeSimilarities(itemIDVector, itemIDRlengths)
    return itemsSimilarities

def generateUsersMeans(userIDDict):
    userMeanDict = {}
    for userID in userIDDict.keys():
        userRatings = userIDDict[userID]
        total = 0.0
        for ratingInfo in userRatings:
            total += ratingInfo['rating']
        mean = total / len(userRatings)
        userMeanDict[userID] = mean
    return userMeanDict

def generateItemsVectors(itemIDDict, itemMeanDict):
    itemIDVector = {}
    itemIDRLengths = {}
    for itemID in itemIDDict.keys():
        itemList = itemIDDict[itemID]
        itemDict = {}
        itemRLength = 0.0
        itemMean = itemMeanDict[itemID]        
        for ratingInfo in itemList:
            userID = ratingInfo['userID']
            rating = ratingInfo['rating']
            itemRLength += (rating - itemMean)**2
            itemDict[userID] = rating - itemMean
        itemRLength = math.sqrt(itemRLength)
        if itemRLength == 0:
            itemRLength = 0.000001
        assert itemRLength > 0
        itemIDRLengths[itemID] = itemRLength
        itemIDVector[itemID] = itemDict
    return itemIDVector, itemIDRLengths


#def computeLength(itemVector):


def computeSimilaritiesWithFileSaving(objectIDVector, objectIDRLengths,
        filePrefix):
    objectSimilarities = {}
    objectIDs = objectIDVector.keys()
    objectIDsLength = len(objectIDs)
    total = objectIDsLength * objectIDsLength / 2
    count = 0
    for i in range(objectIDsLength):
        print i
        objectIDi = objectIDs[i]
        objectiVector = objectIDVector[objectIDi]
        objectiRLength = objectIDRLengths[objectIDi]
        for j in range(i + 1, objectIDsLength):
            objectIDj = objectIDs[j]
            objectjVector = objectIDVector[objectIDj]
            objectjRLength = objectIDRLengths[objectIDj]
            similarity = computeSimilarity(objectiVector, objectjVector, objectiRLength, objectjRLength)
            if similarity < 0.0001:
                similarity = 0.0
            objectSimilarities[objectIDj] = round(similarity,4)
        utils.saveDictToCSVFile(filePrefix+str(objectIDi),objectSimilarities)
    #print ''
    #print len(objectsSimilarities)
    #return dwobjectsSimilarities


def computeSimilarities(objectIDVector, objectIDRLengths):
    objectsSimilarities = {}
    objectIDs = objectIDVector.keys()
    objectIDsLength = len(objectIDs)
    total = objectIDsLength * objectIDsLength / 2
    print total
    count = 0
    for i in range(objectIDsLength):
        objectIDi = objectIDs[i]
        objectiVector = objectIDVector[objectIDi]
        objectiRLength = objectIDRLengths[objectIDi]
        for j in range(i + 1, objectIDsLength):
            objectIDj = objectIDs[j]
            objectjVector = objectIDVector[objectIDj]
            objectjRLength = objectIDRLengths[objectIDj]
            similarity = computeSimilarity(objectiVector, objectjVector, objectiRLength, objectjRLength)
            if objectIDi not in objectsSimilarities:
                objectsSimilarities[objectIDi] = {}
                objectsSimilarities[objectIDi][objectIDj] = similarity
            else:
                objectsSimilarities[objectIDi][objectIDj] = similarity
            if objectIDj not in objectsSimilarities:
                objectsSimilarities[objectIDj] = {}
                objectsSimilarities[objectIDj][objectIDi] = similarity
            else:
                objectsSimilarities[objectIDj][objectIDi] = similarity
            count += 1
            if count % 100000 == 0:
                #print objectsSimilarities
                utils.printProgress(count)
#            print count, total
    #print ''
    print len(objectsSimilarities)
    return objectsSimilarities


def computeSimilarity(itemiVector, itemjVector, itemiRLength, itemjRLength):
    numerator = 0.0
    for userID in itemiVector.keys():
        if userID in itemjVector:
            numerator += itemiVector[userID] * itemjVector[userID]
    if numerator < 0.0001:
        return 0.0
        #    numerator = 0.00001
    similarity = numerator / (itemiRLength * itemjRLength)
    return round(similarity, 4)

def predictionii(itemSimilarities, trainUserIDDict, testUserIDDict, itemMeanDict):
#    threshold = 0.001
#    K = 20
    RMSE = 0
    count = 0

    for userID in testUserIDDict.keys():
        predictionList = testUserIDDict[userID]
        trainList = trainUserIDDict[userID]
        for ratingInfo in predictionList:
            itemIDi = ratingInfo['itemID']
            ratingi = ratingInfo['rating']
            if itemIDi not in itemSimilarities:
                continue
            itemiMean = itemMeanDict[itemIDi]                    
            itemiSimilarityList = itemSimilarities[itemIDi]
            """
            sortedList = sorted(itemiSimilarityList.iteritems(), key=operator.itemgetter(1))
            sortedList.reverse()
            numerator = 0.0
            denominator = 0.0
            for index in range(K):
                if index >= len(sortedList):
                    break
                similarityij = sortedList[index][1]
                denominator += math.abs(similarityij)
            """
            denominator = 0.0
            numerator = 0.0
            for trainRatingInfo in trainList:
                itemIDj = trainRatingInfo['itemID']
                ratingj = trainRatingInfo['rating']
                itemjMean = itemMeanDict[itemIDj]
                if itemIDj not in itemiSimilarityList:
                    continue
                similarityij = itemiSimilarityList[itemIDj]
                denominator += math.fabs(similarityij)
                numerator += (ratingj - itemjMean) * similarityij
            if denominator == 0:
                denominator = 1
            predictedRating = itemiMean + numerator / denominator
            RMSE += (predictedRating - ratingi)**2
            count += 1
    RMSE /= count
    RMSE = math.sqrt(RMSE)
    print 'RMSE: ', RMSE, 'count: ', count

    return RMSE


