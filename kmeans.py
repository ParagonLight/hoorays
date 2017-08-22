import numpy as np
import random
def initSetOfKMeans(K, matrix):
    kmeans = {}
    length = len(matrix)    
    clusters = {}
#    lastValue
    for i in range(K):        
        value = int(random.random() * length)
        while value in kmeans:
            value = int(random.random() * length)
        v = matrix[value].copy()
        kmeans[value] = v
        clusters[kmeans.keys()[0]] = matrix
    #print kmeans
    return kmeans, clusters

def assignmentStep(kmeans, clusters):
    newClusters = {}
    flag = False
    for key in clusters.keys():
        subMatrix = clusters[key]
        for vector in subMatrix:
            shortestDist = 1000
            shortestMeanKey = -1
            for meanKey in kmeans.keys():
                mean = kmeans[meanKey]
                dist = np.linalg.norm(mean - vector)
                if shortestDist > dist:
                    shortestDist = dist
                    shortestMeanKey = meanKey
            if key != shortestMeanKey:
                flag = True
            if shortestMeanKey not in newClusters:
                newClusters[shortestMeanKey] = np.vstack([vector])
            else:
                m = newClusters[shortestMeanKey] 
                #print m.shape, vector.shape
                newClusters[shortestMeanKey] = np.vstack([m, vector])
    return newClusters, flag

def updateStep(kmeans, clusters):
    for key in clusters.keys():
        subMatrix = clusters[key]
        means = subMatrix.sum(axis = 0) / len(subMatrix)
        kmeans[key] = means
        #print means.shape, subMatrix.shape
    return kmeans

def kmeansRuning(matrix, K):
    kmeans, clusters = initSetOfKMeans(K, matrix)
   # print kmeans
    clusteringFlag = True
    count = 1
    while clusteringFlag:
        clusters, clusteringFlag = assignmentStep(kmeans, clusters)
        if clusteringFlag == False:
            break
        kmeans = updateStep(kmeans, clusters)
        print 'Epoch:', count
        #clusteringFlag = False
        count += 1
    return kmeans

def findusersCluster(kmeans, usersVector):
    length = len(usersVector)
    usersCluster = {}
    for userID in range(length):
        usersCluster[userID] = -1

#    usersCluster = {userID:-1 for userID in range(length)}
    for index in range(length):
        userVector = usersVector[index]
        minDist = 1000    
        elector = -1
        for key in kmeans:
            defaultVector = kmeans[key]
            dist = np.linalg.norm(userVector - defaultVector)
            if dist < minDist:
                minDist = dist
                elector = key
        usersCluster[index] = elector
    return usersCluster

