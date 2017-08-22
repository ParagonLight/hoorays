import utils
import datetime
import matrixFactorization as mf
import random
import numpy as np
import copy
def convertTimestampToDateTime(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp))#.strftime('%Y-%m-%d %H:%M:%S')

def convertStringToRatingInfo(string, separator):
    array = string.split(separator)
    userID = array[0]
    itemID = array[1]
    rating = array[2]
    timestamp = array[3]
    return userID, itemID, rating, -1, timestamp

def convertStringToRatingInfoForEachMovie(string, separator):
    array = string.split(separator)
    userID = array[0]
    itemID = array[1]
    rating = array[2]
    weight = array[3]
    timestamp = array[4]
    return userID, itemID, rating, weight, timestamp



def addOneRatingToInterviewSet(interviewSet, evaluationSet):
    interviewCount = 0
    evaluationCount = 0
    count = 0

    for userID in evaluationSet.keys():            
        interviewList = interviewSet[userID]
        evaluationList = evaluationSet[userID]
        if len(evaluationList) <= 1:
            evaluationSet.pop(userID, None)
            interviewSet.pop(userID, None)
            count += 1
            continue

        random.shuffle(evaluationList)
        interviewList.append(evaluationList.pop())
        interviewSet[userID] = interviewList
        evaluationSet[userID] = evaluationList
        interviewCount += len(interviewList)
        evaluationCount += len(evaluationList)
    print 'Total size of interview set is:',interviewCount, 'Total size of evaluation set is:', evaluationCount
    print 'Number of error user: ', count 
    return interviewSet, evaluationSet

def generateInterviewSet(dataSet, K):
    tempSet = {}
    interviewSet = {}
    evaluationSet = {}
    interviewCount = 0
    evaluationCount = 0
    count = 0
    for ratingInfo in dataSet:
        userID = ratingInfo['userID']
        if userID not in tempSet:
            tempSet[userID] = [ratingInfo]
        else:
            tempSet[userID].append(ratingInfo)
    print len(tempSet)
    count = 0
    for userID in tempSet.keys():
        userRatings = tempSet[userID]
        if len(userRatings) <= K:
            count += 1
            continue
        random.shuffle(userRatings)
        interviewList = userRatings[0:K]
        evaluationList = userRatings[K:]
        interviewCount += len(interviewList)
        evaluationCount += len(evaluationList)
        interviewSet[userID] = interviewList
        evaluationSet[userID] = evaluationList
    print 'Total size of interview set is:',interviewCount, 'Total size of evaluation set is:', evaluationCount
    print count
    return interviewSet, evaluationSet

def generateInterviewSetWithActiveLearning(dataSet, K, itemsVector, itemKMeans):
    tempSet = {}
    interviewSet = {}
    evaluationSet = {}
    interviewCount = 0
    evaluationCount = 0
    count = 0
    for ratingInfo in dataSet:
        userID = ratingInfo['userID']
        if userID not in tempSet:
            tempSet[userID] = [ratingInfo]
        else:
            tempSet[userID].append(ratingInfo)
    print len(tempSet)
    count = 0
    for userID in tempSet.keys():
        userRatings = tempSet[userID]
        if len(userRatings) <= K:
            count += 1
            continue
        random.shuffle(userRatings)
        keysMark = {}
        interviewList = []
        evaluationList = []
        kmeansDict = itemKMeans.copy()        
        for ratingInfo in userRatings:
            if len(keysMark) >= K:
                evaluationList.append(ratingInfo)
            else:
                itemID = ratingInfo['itemID']
                itemVector = itemsVector[itemID - 1]
                minDist = 1000    
                elector = -1
                for key in kmeansDict:
                    defaultVector = kmeansDict[key]
                    dist = np.linalg.norm(itemVector - defaultVector)
                    if dist < minDist:
                        elector = key
            #    print elector
                if elector not in keysMark:
                    keysMark[elector] = 1
                    interviewList.append(ratingInfo)
                    del kmeansDict[elector]
                else:                
                    evaluationList.append(ratingInfo)
                    
        interviewCount += len(interviewList)
        evaluationCount += len(evaluationList)
        interviewSet[userID] = interviewList
        evaluationSet[userID] = evaluationList
    print 'Total size of interview set is:',interviewCount, 'Total size of evaluation set is:', evaluationCount
    print count
    return interviewSet, evaluationSet

def splitDatasetWithUsers(splitRatio, ratingFile, dataset, savePrefix):
    dataList = {}
    userIDDict = {}
    itemIDDict = {}
    functions = {'eachMovie6':convertStringToRatingInfoForEachMovie,'eachMovie':convertStringToRatingInfoForEachMovie,'movieLens':convertStringToRatingInfo}
    
    for row in ratingFile:
        userID, itemID, rating, weight, timestamp = functions[dataset](row, separator) 
#        userID, itemID, rating, timestamp = convertStringToRatingInfo(row, separator)
        if int(userID) not in userIDDict:
            userIDDict[int(userID)] = 1
            dataList[int(userID)] = [row]
        else:
            dataList[int(userID)].append(row)
        if int(itemID) not in itemIDDict:
            itemIDDict[int(itemID)] = 1            
#        dataList.append(row)
    
    userIDList = userIDDict.keys()
    random.shuffle(userIDList)
    
    totalLength = len(userIDList)
    splitedIndex = totalLength * splitRatio

    trainSet = utils.createWriteFile(savePrefix + 'splitedUserTrain.dat')
    testSet = utils.createWriteFile(savePrefix + 'splitedUserTest.dat')
    count = 0
    for userID in userIDList:
        rows = dataList[userID]
        if userID > splitedIndex:
            for row in rows:
                testSet.write(row)
        else:
            for row in rows:
                trainSet.write(row)
            count += 1
    print count, len(userIDList) - count



def generateDataset(ratingFile,separator, dataset):
    dataList = []
    userIDDict = {}
    itemIDDict = {}
    functions = {'eachMovie6':convertStringToRatingInfoForEachMovie,'eachMovie':convertStringToRatingInfoForEachMovie,'movieLens':convertStringToRatingInfo, 'netflex':convertStringToRatingInfo}        
    for row in ratingFile:
        userID, itemID, rating, weight, timestamp = functions[dataset](row, separator)
        ratingInfo = {}
        if dataset == 'eachMovie':
            if rating == '0.00':
                continue
            rating = float(rating) * 5
            if rating == 0:
                continue
        if dataset == 'eachMovie6':
            rating = float(rating) * 5 + 1
        ratingInfo['rating'] = float(rating)
        ratingInfo['itemID'] = int(itemID)
        ratingInfo['userID'] = int(userID)
        ratingInfo['timestamp'] = timestamp
        if int(userID) not in userIDDict:
            userIDDict[int(userID)] = [ratingInfo]
        else:
            userIDDict[int(userID)].append(ratingInfo)
        if int(itemID) not in itemIDDict:
            itemIDDict[int(itemID)] = [ratingInfo]
        else:
            itemIDDict[int(itemID)].append(ratingInfo)
        dataList.append(ratingInfo)
    return dataList, userIDDict, itemIDDict



def generateNewTestSet(evaluationSet):
    dataList = []
    userIDDict = {}
    itemIDDict = {}
    for userID in evaluationSet.keys():
        for ratingInfo in evaluationSet[userID]:
            userID = ratingInfo['userID']
            itemID = ratingInfo['itemID']
            if int(userID) not in userIDDict:
                userIDDict[int(userID)] = [ratingInfo]
            else:
                userIDDict[int(userID)].append(ratingInfo)
            if int(itemID) not in itemIDDict:
                itemIDDict[int(itemID)] = [ratingInfo]
            else:
                itemIDDict[int(itemID)].append(ratingInfo)
            dataList.append(ratingInfo)
    return dataList, userIDDict, itemIDDict


def combineTwoSets(interviewSet, trainSet, trainUserIDDict, trainItemIDDict):
    dataList = copy.deepcopy(trainSet)
    userIDDict = copy.deepcopy(trainUserIDDict)
    itemIDDict = copy.deepcopy(trainItemIDDict)
    for userID in interviewSet.keys():
        for ratingInfo in interviewSet[userID]:
            userID = ratingInfo['userID']
            itemID = ratingInfo['itemID']
            if int(userID) not in userIDDict:
                userIDDict[int(userID)] = [ratingInfo]
            else:
                userIDDict[int(userID)].append(ratingInfo)
            if int(itemID) not in itemIDDict:
                itemIDDict[int(itemID)] = [ratingInfo]
            else:
                itemIDDict[int(itemID)].append(ratingInfo)
            dataList.append(ratingInfo)
    return dataList, userIDDict, itemIDDict

def parseRatingInfoWithTimestampKey(ratingFile, separator):
    timestampArray = []
    ratingDict = {} #key is timestamp
    for row in ratingFile:
        userID, itemID, rating, timestamp = convertStringToRatingInfo(row, separator)        
        timestampArray.append(int(timestamp))
        ratingDict[int(timestamp)] = row
    return timestampArray, ratingDict

def showNewUserRatingWithTime(timestampArray, ratingDict, separator): 
    oldUserDict = {}
    oldItemDict = {}
    count = 0

    for timestamp in timestampArray:
        userID, itemID, rating, timestamp = convertStringToRatingInfo(ratingDict[timestamp], separator)
        if userID not in oldUserDict:
            oldUserDict[userID] = 1
        else:
            oldUserDict[userID] += 1
        if itemID not in oldItemDict:
            oldItemDict[itemID] = 1
        else:
            oldItemDict[itemID] += 1
        count += 1
        print 'remaining new user:', userNum - len(oldUserDict),'ratio:',len(oldUserDict)/float(userNum), 'remaining new item:', movieNum - len(oldItemDict),'ratio:',len(oldUserDict)/float(userNum), 'remaining ratings:', 1000209 - count, timestamp
   
def randomGenerateDataSet(ratingFile):
    trainSet = utils.createWriteFile('RDTrain.dat')
    testSet = utils.createWriteFile('RDTest.dat')

    for row in ratingFile:
        random.seed()
        value = random.random()
        if value > 0.75:
            testSet.write(row)
        else:
            trainSet.write(row)


def randomSplitDatasetWithRatio(dataSet, ratio):
    setOne = []
    setTwo = []
    splitIndex = int(len(dataSet) * ratio)
    random.shuffle(dataSet)
    setOne = dataSet[0:splitIndex]
    setTwo = dataSet[splitIndex:]
    return setOne, setTwo



def splitDataWithACertainTime(timestampArray, ratingDict, separator, splitRatio):
    oldUserDict = {}
    oldItemDict = {}
    count = 0
    trainFile = utils.createWriteFile('train.dat')
    testFile = utils.createWriteFile('test.dat')
    total = len(timestampArray)
    ratingsWithNewUser = 0
    ratingsWithNewItem = 0
    ratings = 0
    newToNew = 0
    for timestamp in timestampArray:
        string = ratingDict[timestamp]
        userID, itemID, rating, timestamp = convertStringToRatingInfo(string, separator)
        if count <= split * total:
            if userID not in oldUserDict:
                oldUserDict[userID] = 1
            else:
                oldUserDict[userID] += 1
            if itemID not in oldItemDict:
                oldItemDict[itemID] = 1
            else:
                oldItemDict[itemID] += 1
            trainFile.write(string)
        else:
            flag = 0
            ratings += 1
            if userID not in oldUserDict:
                ratingsWithNewUser += 1
                flag = 1
            if itemID not in oldItemDict:
                ratingsWithNewItem += 1
                if flag == 1:
                    newToNew += 1

            testFile.write(string)
        count += 1

    print ratings, ratingsWithNewUser, ratingsWithNewItem, newToNew



if __name__ == '__main__':
    movieNum = 3883
    userNum = 6040
    separator = '::'
    split = 0.75
    #ratingFileName = 'ml-1m/ratings.dat'
    ratingFileName = 'ctrdata/elect/elect.dat'
    savePrefix = 'ctrdata/'
    dataset = 'movielens'

    ratingFile = utils.createReadFile(ratingFileName)
    splitDatasetWithUsers(split, ratingFile, dataset, savePrefix)
    
#    randomGenerateDataSet(ratingFile)

#    timestampArray, ratingDict = parseRatingInfoWithTimestampKey(ratingFile, separator)
#    timestampArray.sort()
    
#    splitDataWithACertainTime(timestampArray, ratingDict, separator, split)

