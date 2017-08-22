import csv
#from tempfile import 
import numpy as np
import sys

def saveNumpyMatrix(fileName, matrix):
    np.save(fileName, matrix)

def loadNumpyMatrix(fileName):
    return np.load(fileName)

def saveNumpyVector(fileName, vector):
    np.save(fileName,vector)

def loadNumpyVector(fileName):
    return np.load(fileName)


def createReadCSVFile(fileName):
    return csv.reader(open(fileName,'rb'))

def createWriteCSVFile(fileName):
    return csv.writer(open(fileName,'wb'))

def createReadFile(fileName):
    return open(fileName,'r')

def createWriteFile(fileName):
    return open(fileName,'w')


def saveSet(dataSet, filename):
    count = 0
    f = createWriteFile(filename)
    for userID in dataSet.keys():
        f.write('userID,' + str(userID) + '\n')
        ratingList = dataSet[userID]
        lines = []
        for ratingInfo in ratingList:
            rating = str(ratingInfo['rating'])
            itemID = str(ratingInfo['itemID'])
            userID = str(ratingInfo['userID'])
            timestamp = ratingInfo['timestamp']
            line = rating + ',' + itemID + ',' + userID + ',' + timestamp
            lines.append(line)
            count += 1
        f.writelines(lines)
    print 'total count for ', filename, ' is:',count

def loadSet(filename):
    count = 0
    f = createReadFile(filename)
    dataSet = {}
    lastUserID = '-1'
    ratingList = []
    
    for row in f:
        array = row.split(',')
        if array[0] == 'userID':
            if lastUserID != '-1':
                dataSet[lastUserID] = ratingList
            userID = array[1]
            userID = userID.replace('\n','')
            ratingList = []
            lastUserID = userID
            #print userID, array
        else:
            ratingInfo = {}
            ratingInfo['rating'] = float(array[0])
            ratingInfo['itemID'] = int(array[1])
            ratingInfo['userID'] = int(array[2])
            ratingInfo['timestamp'] = array[3]
            ratingList.append(ratingInfo)
            count += 1
    dataSet[lastUserID] = ratingList
    
    print 'total count for ', filename, ' is:',count
    return dataSet
        
    
def saveDictToCSVFile(fileName, dictionary):
    w = csv.writer(open(fileName, 'w'))
    np.savetxt(fileName, dictionary.values(), delimiter=",")
    
#    for key, val in dictionary.items():
#        w.writerow(np.vstack[key, val])

def loadDictFromCSVFile(fileName):
    rows = np.loadtxt(fileName, dtype='float', delimiter=',')
    dictionary = {}
    count = 0
    for row in rows:
#        key = row[0]
#        value = row[1:]
        value = np.array(row)
        dictionary[count] = value
        count += 1
    return dictionary

def countZeroRating(dataSet):
    for key in dataSet.keys():
        dataList = dataSet[key]
        newList = []
        count = 0
        for ratingInfo in dataList:
            rating = int(ratingInfo['rating'])
            if rating == 0:
                count += 1
            else:
                newList.append(ratingInfo)
        dataSet[key] = newList
        assert len(dataList) == (len(newList) + count)
    print count
    return dataSet



def generateUserVectorForItem(itemSet, usersVector):
    usersVectorForItem = {}
    itemKeys = itemSet.keys()
    print '1'
    for key in itemKeys:
        ratings = itemSet[key]
        count = 0
        
        userVector = np.zeros(usersVector[0].shape)
        for ratingInfo in ratings:
            userID = ratingInfo['userID']
            userVector += usersVector[int(userID) - 1]
        userVector /= len(ratings)
        usersVectorForItem[key] = userVector
    return usersVectorForItem 


def generateUserVectorForItem(itemSet, usersVector):
    usersVectorForItem = {}
    itemKeys = itemSet.keys()
    print '2'
    for key in itemKeys:
        ratings = itemSet[key]
        count = 0
        vectorsSet = {}
        for r in range(1,7):
            userVector = np.zeros(usersVector[0].shape)
            vectorsSet[r] = userVector
        ratingSet = {1:0,2:0,3:0,4:0,5:0,6:0}
        for ratingInfo in ratings:
            userID = ratingInfo['userID']
            rating = int(ratingInfo['rating'])
            vectorsSet[rating] += usersVector[int(userID) - 1]
            ratingSet[rating] += 1
        for r in range(1,7):
            if ratingSet[r] > 0:
                vectorsSet[r] /= ratingSet[r]
        usersVectorForItem[key] = vectorsSet
    return usersVectorForItem 



def countRatings(itemSet):
    itemCountedSet = {}
    itemKeys = itemSet.keys()
    for key in itemKeys:
        ratings = itemSet[key]
        count = 0
        #ratingSet = {1:0,2:0,3:0,4:0,5:0,6:0}
        ratingSet = {0.1:0, 0.4:0, 0.9:0, 0.7:0,0.01:0,0.5:0,1:0,1.5:0,2:0,2.5:0,3:0,3.5:0,4:0,4.5:0,5:0}
        for ratingInfo in ratings:
            ratingj = float(ratingInfo['rating'])
            ratingSet[ratingj] += 1
        itemCountedSet[key] = ratingSet
    return itemCountedSet

"""
output format:
    return a list, which contains lists of (x,y) pair case, and a value represents the number of features 
    for each case(x,y), the format is as follow:
        [y, xVector]
        xVector[index] = value
    Due to the sparsity, we use dict to store the vector.
"""
def loadingInputFile(inputFileName, fileType):
    indexLength = 0
    inputFile = createReadCSVFile(inputFileName)
    inputStruct = []
    for row in inputFile:
        rowList = row[0].split(' ')
        y = float(rowList[0])
        eleDict = {}
        for ele in rowList[1:]:
        #    print ele
            splitedEle = ele.split(':')
            index = int(splitedEle[0])
         #   print index, indexLength
            
            if index > indexLength:
                indexLength = index
            value = float(splitedEle[1])
            eleDict[index] = value
        structedRow = [y,eleDict]
        inputStruct.append(structedRow)
    print 'Finish loading',fileType,'file, the number of elements is',len(inputStruct)
    
    return inputStruct, indexLength+1
        
    
def generatePairwiseVector(inputStruct, indexLength):
    pairwiseVector = []
    for row in inputStruct:
        vector = row[1]
        keys = sorted(vector.keys())
        keyLength = len(keys)
        length = indexLength
        pairwiseRow = {}
        index = 0

        for i in range(keyLength):
            for j in range(i+1,keyLength):
                pairwiseRow[index + j] = vector[keys[i]] * vector[keys[j]]
            index += length
            length -= 1
        pairwiseVector.append(pairwiseRow)
    return pairwiseVector, index
        
def printProgress(current):
    sys.stdout.write('\r')
    sys.stdout.write("%d"% current) 
    sys.stdout.flush()
