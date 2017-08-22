import dataAnalysis as da
import utils
import matrixFactorization as mf 
import rapare as ra
import sys
import time
import ctrUtil as cu


def paramTuning(argv):
    alphas = [0.01]
    Lambdaus = [0.5]
    Lambdavs = [0.1]
    Lambdaus = [0.5]
    Lambdavs = [0.5]
    Lambdauvs = [0.01]
    kernels = ['sigmoid']
    initFunctions = ['random']
    features = [200]
    epochs = 201 
    data = 'ml'
    data = argv[0]
    rootPath = 'ctrdata/'+data +'/'
    tenPlus = False 
    if tenPlus:
        p = '-10plus'
    else:
        p = ''
    tru = rootPath + data + '-Train-User'+p+'.dat'
    tri = rootPath + data + '-Train-Item'+p+'.dat'
    teu = rootPath + data + '-Test-User'+p+'.dat'
    tei = rootPath + data + '-Test-Item'+p+'.dat'
    theta = rootPath + 'final.gamma'
    for kernel in kernels:
        for initFunction in initFunctions:
            for feature in features:
                thetaPath = theta + str(feature)
                argvs = [data, kernel, initFunction, feature, tru, tri, teu, tei, thetaPath, rootPath]
                for alpha in alphas:
                    for Lambdau in Lambdaus:
                        for Lambdav in Lambdavs:
                            for Lambdauv in Lambdauvs:
                                print "##############################"
                                print "alpha:",alpha," lambdau:", Lambdau, "lambdav:", Lambdav, "lambdauv", Lambdauv
                                solve(argvs, alpha, Lambdau, Lambdav, Lambdauv, epochs) 

    
def solve(argv, alpha, Lambdau, Lambdav, Lambdauv, epochs):
    print argv
    MF = True 
    CTR = False
    RAPMF = True
    RAPARE = False
    hasLDA = False 
    save = True 
    dataset = argv[0]
    if dataset == 'delicious' or dataset == 'fm' or dataset == 'blog':
        scale = 1
    else:
        scale = 5
    kernel = argv[1]
    initFunction = argv[2]
    featureK = argv[3]
    tru = argv[4]
    tri = argv[5]
    teu = argv[6]
    tei = argv[7]
    topN = [1,5,10,15,20,25,30,35,40,45,50]
    thetaPath = argv[8]
    filePrefix = argv[9]

    print 'load train data...'
    trainUser, trainData = cu.loadData(tru, 'user')
    trainItem, noUse= cu.loadData(tri, 'item')
    userNum = len(trainUser.keys())
    itemNum = len(trainItem.keys())
    noUse = None

    print 'load test data...'
    testUser, testData = cu.loadData(teu, 'user')
    noUse = None

    print 'User No.', userNum, 'Item No.', itemNum, 'ratings in Train:',len(trainData), 'rating in Test:', len(testData)

    mean = 0.0
    theta = []
    if RAPARE or RAPMF:
        if RAPARE:
            print 'load theta...'
            theta = cu.loadTheta(thetaPath)
        else:
            theta = []
        usersVector = None 
        itemsVector = None
        usersBias = None
        itemsBias = None
        usersRatingCountedSet = utils.countRatings(trainUser)
        itemsRatingCountedSet = utils.countRatings(trainItem)
        ra.train(mean, userNum, itemNum, featureK, trainData, testData, epochs,
                alpha, Lambdau, dataset, usersRatingCountedSet,
                itemsRatingCountedSet, True, usersVector, itemsVector, usersBias,
                itemsBias, Lambdauv, theta, filePrefix + str(featureK), save, scale, Lambdav, hasLDA, tru,tri,teu,topN)


if __name__ == '__main__':
    argv = sys.argv[1:]
    #dataset = argv[0]
    paramTuning(argv)
