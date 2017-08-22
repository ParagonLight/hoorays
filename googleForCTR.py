import dataAnalysis as da
import utils
import matrixFactorization as mf 
import eloModel as em
import rapare as ra
import sys
import time
import ctrUtil as cu


def paramTuning(argv):
#    alphas = [0.005,0.001,0.01,0.1]
    alphas = [0.01]
#    Lambdas = [0.5,0.4,0.01,0.001]
    Lambdaus = [0.5]#,0.05,0.1,0.5,1]#,0.05,0.1,0.001]
    Lambdavs = [0.5]#,0.05,0.1,0.5,1]#,0.05,0.1,0.001]
    Lambdauvs = [0.01]
    kernels = ['sigmoid']
    initFunctions = ['`random']
    features = [200]
    types = ['item']
    epochs = 200
    data = 'google75'
    rootPath = 'ctrdata/'+data+'/'
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
            for t in types:
                for feature in features:
                    thetaPath = theta + str(feature) 
                    argvs = [data, t, kernel, initFunction, feature, tru, tri, teu, tei, thetaPath, rootPath]
                    for alpha in alphas:
                        for Lambdau in Lambdaus:
                            for Lambdav in Lambdavs:
                                for Lambdauv in Lambdauvs:
                                    print "##############################"
                                    print "alpha:",alpha," lambdau:", Lambdau, "lambdav:", Lambdav, "lambdauv", Lambdauv, "thetaPath:", thetaPath
                                    solve(argvs, alpha, Lambdau, Lambdav, Lambdauv, epochs) 
    
    
def solve(argv, alpha, Lambdau, Lambdav, Lambdauv, epochs):
    print argv
    MF = False
    CTR = False
    RAPMF = False
    RAPARE = True 
    hasLDA = False 
    save = True 
    dataset = argv[0]
    if dataset == 'delicious' or dataset == 'fm':
        scale = 1
    else:
        scale = 5
    newElement = argv[1]    
    kernel = argv[2]
    initFunction = argv[3]
    featureK = argv[4]
    tru = argv[5]
    tri = argv[6]
    teu = argv[7]
    tei = argv[8]
    topN = [1,5,10,15,20,25,30,35,40,45,50]
    thetaPath = argv[9]
    filePrefix = argv[10]
    print 'feature:', featureK
    print 'load train data...'
    trainUser, trainData = cu.loadData(tru, 'user')
    trainItem, noUse= cu.loadData(tri, 'item')
    userNum = len(trainUser.keys())
    itemNum = len(trainItem.keys())
    noUse = None

    print 'load test data...'
    testUser, testData = cu.loadData(teu, 'user')
    #testItem, testDataNoUse = cu.loadData(tei, 'item')
    noUse = None

    print 'User No.', userNum, 'Item No.', itemNum, 'ratings in Train:',len(trainData), 'rating in Test:', len(testData)

    #mean = 4.31725404304#video
    #mean = 4.20794266709#google play
    mean = 4.18519626023#auto
    mean = 0.0
    theta = []
    if MF:
        print 'start matrix factorization'
        mf.train(userNum, itemNum, featureK, trainData, testData, epochs, Lambdau, Lambdav, alpha, mean, filePrefix + str(featureK), save, theta, scale)
    if CTR:
        print 'load theta from...', thetaPath
        theta = cu.loadTheta(thetaPath)
        print 'start CTR'
        mf.train(userNum, itemNum, featureK, trainData, testData, epochs, Lambdau, Lambdav, alpha, mean, filePrefix + str(featureK), save, theta, scale)


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
