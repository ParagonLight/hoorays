import re
import nltk
import random
import json
import gzip

def parse(path):
    print path
    #g = gzip.open(path, 'r')
    g = open(path, 'r')
    for l in g:
        yield eval(l)


def discardUsers(U, V, threshold):
    pass


def preCTRUserAndItem(inputFile, outputPre):
    '''
    fuTr = open(outputPre+'-Train-User-10plus.dat', 'w')
    fiTr = open(outputPre+'-Train-Item-10plus.dat', 'w')
    fuTe = open(outputPre+'-Test-User-10plus.dat', 'w')
    fiTe = open(outputPre+'-Test-Item-10plus.dat', 'w')
    '''
    fuTr = open(outputPre+'-Train-User.dat', 'w')
    fiTr = open(outputPre+'-Train-Item.dat', 'w')
    fuTe = open(outputPre+'-Test-User.dat', 'w')
    fiTe = open(outputPre+'-Test-Item.dat', 'w')
    
    userInt = 0
    itemInt = 0
    cache = []
    count = 0
    totalUser = {}
    userID = {}
    itemID = {}
    userItem = {}
    itemUser = {}
    userItemTe = {}
    itemUserTe = {}
    print userInt
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
        if u in totalUser:
            totalUser[u].append(str(itemID[item])+':'+str(rating))
        else:
            totalUser[u] = [str(itemID[item])+':'+str(rating)]
        '''
        if tag == 'train':
            if u in userItem:
                userItem[u].append(str(itemID[item])+':'+str(rating))
            else:
                userItem[u] = [str(itemID[item])+':'+str(rating)]
            if i in itemUser:
                itemUser[i].append(str(userID[user])+':'+str(rating))
            else:
                itemUser[i] = [str(userID[user])+':'+str(rating)]
        else:
            if u in userItemTe:
                userItemTe[u].append(str(itemID[item])+':'+str(rating))
            else:
                userItemTe[u] = [str(itemID[item])+':'+str(rating)]
            if i in itemUserTe:
                itemUserTe[i].append(str(userID[user])+':'+str(rating))
            else:
                itemUserTe[i] = [str(userID[user])+':'+str(rating)]
        '''
        count += 1
        if count % 10000 == 0:
            print count
    totalItem = {}
    newTotalUser = {}
    for u in totalUser.keys():
        rs = totalUser[u]
        if len(rs) < 10:
            continue
        newTotalUser[u] = rs
        for r in rs:
            random.seed()
            value = random.random()
            if value < 0.75:
                tag = 'train'
            else:
                tag = 'test'
            strs = r.split(':')
            iid = int(strs[0])
            rating = strs[1]
            if iid in totalItem:
                totalItem[iid].append(str(u)+':'+rating)
            else:
                totalItem[iid] = [str(u)+':'+rating]
            if tag == 'train':
                if u in userItem:
                    userItem[u].append(str(iid)+':'+rating)
                else:
                    userItem[u] = [str(iid)+':'+rating]
                if iid in itemUser:
                    itemUser[iid].append(str(u)+':'+rating)
                else:
                    itemUser[iid] = [str(u)+':'+rating]
            else:
                if u in userItemTe:
                    userItemTe[u].append(str(iid)+':'+rating)
                else:
                    userItemTe[u] = [str(iid)+':'+rating]
                if i in itemUserTe:
                    itemUserTe[i].append(str(u)+':'+rating)
                else:
                    itemUserTe[i] = [str(u)+':'+rating]

    print userInt, itemInt
    cache = []
    for u in range(userInt):
        if u not in userItem:
            line = "0"
        else:
            items = userItem[u]
            line = ' '.join(items)
            line = str(len(items)) + ' ' + line
        cache.append(line + '\n') 
    fuTr.writelines(cache)
    cache = []
    for i in range(itemInt):
        if i not in itemUser:
            line = "0"
        else:
            users = itemUser[i]
            line = ' '.join(users)
            line = str(len(users)) + ' ' + line
        cache.append(line + '\n') 
    fiTr.writelines(cache)


    cache = []
    for u in range(userInt):
        if u not in userItemTe:
            line = "0"
        else:
            items = userItemTe[u]
            line = ' '.join(items)
            line = str(len(items)) + ' ' + line
        cache.append(line + '\n') 
    fuTe.writelines(cache)
    cache = []
    for i in range(itemInt):
        if i not in itemUserTe:
            line = "0"
        else:
            users = itemUserTe[i]
            line = ' '.join(users)
            line = str(len(users)) + ' ' + line
        cache.append(line + '\n') 
    fiTe.writelines(cache)

def preCiao(inputFile, outputPre):
    fo = open(outputPre+'Rating.dat', 'w')
    fv = open(outputPre+'Vec.dat', 'w')
    fu = open(outputPre+'UserID.dat', 'w')
    fi = open(outputPre+'ItemID.dat', 'w')
    fd = open(outputPre+'.dat', 'w')
    f = open(inputFile, 'r')
    import re
    from nltk.stem.lancaster import LancasterStemmer
    stemmer = nltk.stem.porter.PorterStemmer()
    stopword = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[A-Za-z]+')
    globalDict = nltk.FreqDist()
    st = LancasterStemmer()
    userInt = 0
    itemInt = 0
    cache = []
    count = 0
    userID = {}
    itemID = {}
    vec = {}
    wId = 0
    reviews = {}
    iiii = 0
    for line in f:
        if iiii < 300000:
            iiii += 1
            continue
        iiii += 1
        
        if line[0] > '9' or line[0] < '0':
            continue
        review = line.strip()
        review = review.split('::::')
        offset = 0
        try:
            #while review[3+offset] < '10' or review[3+offset] > '50':
            while review[3+offset] != '10' and review[3+offset] != '20' and review[3+offset] != '30' and review[3+offset] != '40' and review[3+offset] != '50':
                offset += 1
                if (offset+3) == len(review):
                    break
        except:
            print line
        if (offset+3) == len(review):
            continue
        rating = review[3+offset] 
        rating = float(rating)/10.0
        rating = str(rating)
        user = review[0]
        item = review[1]
        item = unicode(item, "utf-8")
        try:
            reText = review[6+offset]
        except:
            print line
            continue
        #soup = BeautifulSoup(reText)
        
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
        reText = tokenizer.tokenize(reText)
        
        # stopwords, short words
        words = [token.lower() for token in reText if token.lower() not in stopword and len(token) > 2]
        words = [stemmer.stem(ss) for ss in words]
        #words = re.compile('\w+').findall(reText)
        if i in reviews:
            ww = reviews[i]
        else:
            ww = []

        for w in words:
            ww.append(w)
            if w not in vec:
                vec[w] = wId
                wId += 1
        reviews[i] = ww
        if count % 10000 == 0:
            if len(cache) > 0:
                fo.writelines(cache)
                fo.flush()
                cache = []
            if len(cache) > 0:
                fo.writelines(cache)
                fo.flush()
                cache = []
            #if count == 20000:
            #    break
            print count
    fo.writelines(cache)
    count = 0
    cache = []
    for v in vec.keys():
        cache.append(v+'\n')
    fv.writelines(cache)
    fv.flush()
    cache = []
    for u in userID.keys():
        cache.append(u+' '+str(userID[u])+'\n')
    fu.writelines(cache)
    fu.flush()
    cache = []
    for i in itemID.keys():
        cache.append(i+' '+str(itemID[i])+'\n')
    print cache
    fi.writelines(cache)
    fi.flush()
    cache = []
    for i in range(itemInt):
        ww = reviews[i]
        M = 0
        wDic = {}
        for w in ww:
            if w not in wDic:
                wDic[w] = 1 
            else:
                wDic[w] += 1
        M = len(wDic.keys())
        strs = [str(vec[w])+':'+str(wDic[w]) for w in wDic.keys()]
        line = str(M) + ' '
        subLine = ' '.join(strs)
        line = line + subLine
        count += 1
        cache.append(line + '\n')
        if count % 10000 == 0:
            fd.writelines(cache)
            fd.flush()
            cache = []
            print count
    fd.writelines(cache)

def preGZFile1(inputFile, outputPre):
    fo = open(outputPre+'Rating.dat', 'w')
    fv = open(outputPre+'Vec.dat', 'w')
    fu = open(outputPre+'UserID.dat', 'w')
    fi = open(outputPre+'ItemID.dat', 'w')
    fd = open(outputPre+'.dat', 'w')
    import re
    from nltk.stem.lancaster import LancasterStemmer
    stemmer = nltk.stem.porter.PorterStemmer()
    stopword = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[A-Za-z]+')
    globalDict = nltk.FreqDist()
    st = LancasterStemmer()
    userInt = 0
    itemInt = 0
    cache = []
    count = 0
    userID = {}
    itemID = {}
    vec = {}
    wId = 0
    reviews = {}
    for review in parse(inputFile):
        rating = review["overall"]
        user = review['reviewerID']
        item = review['asin']
        reText = review['reviewText']
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
        reText = tokenizer.tokenize(reText)
        
        # stopwords, short words
        words = [token.lower() for token in reText if token.lower() not in stopword and len(token) > 2]
        words = [stemmer.stem(ss) for ss in words]
        #words = re.compile('\w+').findall(reText)
        if i in reviews:
            ww = reviews[i]
        else:
            ww = []

        for w in words:
            ww.append(w)
            if w not in vec:
                vec[w] = wId
                wId += 1
        reviews[i] = ww
        if count % 10000 == 0:
            if len(cache) > 0:
                fo.writelines(cache)
                fo.flush()
                cache = []
            if len(cache) > 0:
                fo.writelines(cache)
                fo.flush()
                cache = []
            #if count == 20000:
            #    break
            print count
    fo.writelines(cache)
    count = 0
    cache = []
    for v in vec.keys():
        cache.append(v+'\n')
    fv.writelines(cache)
    fv.flush()
    cache = []
    for u in userID.keys():
        cache.append(u+' '+str(userID[u])+'\n')
    fu.writelines(cache)
    fu.flush()
    cache = []
    for i in itemID.keys():
        cache.append(i+' '+str(itemID[i])+'\n')
    fi.writelines(cache)
    fi.flush()
    cache = []
    for i in range(itemInt):
        ww = reviews[i]
        M = 0
        wDic = {}
        for w in ww:
            if w not in wDic:
                wDic[w] = 1 
            else:
                wDic[w] += 1
        M = len(wDic.keys())
        strs = [str(vec[w])+':'+str(wDic[w]) for w in wDic.keys()]
        line = str(M) + ' '
        subLine = ' '.join(strs)
        line = line + subLine
        count += 1
        cache.append(line + '\n')
        if count % 10000 == 0:
            fd.writelines(cache)
            fd.flush()
            cache = []
            print count
    fd.writelines(cache)

def preFile(inputFile, outputFile):
    #inputFile = 'food/foods.txt'
    #outputFile = 'food/data.csv'
    inputFile = 'movie/movies.txt'
    outputFile = 'movie/data.csv'
    fi = open(inputFile, 'r')
    fo = open(outputFile, 'w')
    userID = {}
    itemID = {}
    userInt = 0
    itemInt = 0
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


def wIdPlusOne(fileName):
    fw = open(fileName,'r')
    fr = open(fileName+'.new','w')
    cache = []
    for line in fw:
        strs = line.strip().split(' ')
        ws = strs[1:]
        ww = []
        for w in ws:
            wss = w.split(':')
            ww.append(str(int(wss[0])-1)+':'+wss[1])
        newLine = ' '.join(ww)
        newLine = strs[0] + ' ' + newLine + '\n'
        cache.append(newLine)
    fr.writelines(cache)


if __name__ == "__main__":
    #inputFile = "elect/reviews_Home_and_Kitchen.json.gz2" 
    #inputFile = "ctrdata/elect/reviews_Electronics.json.gz2"
    #outputPre = "ctrdata/elect/elect"
    #inputFile = "ctrdata/auto/reviews_Automotive.json"
    #outputPre = "ctrdata/auto/auto"
    #inputFile = "ctrdata/video/reviews_Amazon_Instant_Video.json"
    #outputPre = "ctrdata/video/video"
    inputFile = "ctrdata/elect/elect.json"
    outputPre = "ctrdata/elect/elect"
    #inputFile = "ctrdata/ciao/rating.txt"
    #outputPre = "ctrdata/ciao/ciao"
    preGZFile1(inputFile, outputPre)
    #preCiao(inputFile, outputPre)
    preCTRUserAndItem(inputFile, outputPre)
    #wIdPlusOne(outputPre+'.dat')
    
