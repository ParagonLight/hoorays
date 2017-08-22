import matrixFactorization as mf
import math
import ctrUtil as cu
import utils


def computeMAE(testData, userVectors,itemVectors):
    mae = 0.0
    for row in testData:
        userID = int(row['userID'])
        itemID = int(row['itemID'])
        rating = float(row['rating'])
        # Since list index in python starts from 0 and user ID 
        #in MovieLens dataset start from 1, all IDs should minus one.
        userVector = userVectors[userID - 1]
        itemVector = itemVectors[itemID - 1]
        predictedRating = mf.predictRating(userVector, itemVector, 0,0,0, 5)
        error = mf.computeError(rating, predictedRating)
        mae += abs(error)

    print mae/len(testData)

if __name__ == '__main__':
    MF = True
    CTR = False
    model = False
    featureK = 50
    data = 'google75'
    rootPath = 'ctrdata/'+data +'/'

    teu = rootPath + data + '-Test-User.dat'
    testData, testData1 = cu.loadData(teu, 'user')

    filePrefix = 'ctrdata/google75/500.05_0.5_0.01/RaP_'
    filePrefix = 'ctrdata/google75/2000.1_0.1/MF_' 
    #filePrefix = 'ctrdata/google75/2000.5_0.5_0.01/noLDA_RaP_'
    usersVectorFile = filePrefix + 'usersVector.npy'
    itemsVectorFile = filePrefix + 'itemsVector.npy'
    userVectors = utils.loadNumpyMatrix(usersVectorFile)
    itemVectors = utils.loadNumpyMatrix(itemsVectorFile)
    usersBias = None#utils.loadNumpyMatrix(usersBiasFile)
    itemsBias = None#utils.loadNumpyMatrix(itemsBiasFile)
    computeMAE(testData1, userVectors,itemVectors)
