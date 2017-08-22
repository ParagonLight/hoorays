import re
import json
import gzip

def parse(path):
    print path
    #g = gzip.open(path, 'r')
    g = open(path, 'r')
    for l in g:
        yield eval(l)

def preGZFile(inputFile, outputFile):
    fo = open(outputFile, 'w')
    userInt = 1
    itemInt = 1
    cache = []
    count = 0
    userID = {}
    itemID = {}
    for review in parse(inputFile):
        rating = review["overall"]
        user = review['reviewerID']
        item = review['asin']
        if user in userID:
            u = userID[user]
        else:
            u = userInt
            userID[user] = u
            userInt += 1
        if item in itemID:
            i = itemID[item]
        else:
            i = itemInt
            itemID[item] = i
            itemInt += 1
        line = [str(u), str(i), str(rating), '2015']
        cache.append('::'.join(line) + '\n')
        count += 1
        if count % 10000 == 0:
            fo.writelines(cache)
            fo.flush()
            cache = []
            print count
    fo.writelines(cache)

def preFile(inputFile, outputFile):
    #inputFile = 'food/foods.txt'
    #outputFile = 'food/data.csv'
    inputFile = 'movie/movies.txt'
    outputFile = 'movie/data.csv'
    fi = open(inputFile, 'r')
    fo = open(outputFile, 'w')
    userID = {}
    itemID = {}
    userInt = 1
    itemInt = 1
    cache = []
    count = 0
    prog = re.compile("review/score:*")
    flag1 = 1
    while flag1 == 1:
        flag = 1
        obj = []
        while flag:
            line = fi.readline()
            if line == "\n":
                flag = 0
            obj.append(line.strip())
            if prog.match(line):
                rating = line.strip().split(' ')[1]
        if len(obj) < 4:
            print flag1
            flag1 = 0
            break
        user = obj[1].split(' ')[1]
        item = obj[0].split(' ')[1]
        if user in userID:
            u = userID[user]
        else:
            u = userInt
            userID[user] = u
            userInt += 1
        if item in itemID:
            i = itemID[item]
        else:
            i = itemInt
            itemID[item] = i
            itemInt += 1
        line = [str(u), str(i), str(rating), '2015']
        cache.append('::'.join(line) + '\n')
        count += 1
        if count % 10000 == 0:
            fo.writelines(cache)
            fo.flush()
            cache = []
            print count
    fo.writelines(cache)



if __name__ == "__main__":
    inputFile = "elect/reviews_Home_and_Kitchen.json.gz2" 
    inputFile = "elect/reviews_Electronics.json.gz2"
    outputFile = "elect/hk.dat"
    preGZFile(inputFile, outputFile)

    
