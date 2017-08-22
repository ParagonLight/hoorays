import os
import nltk
from math import log
import codecs
import time
import datetime

wordsByFrequency = open("words-by-frequency.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(wordsByFrequency)))) for i,k in enumerate(wordsByFrequency))
maxword = max(len(x) for x in wordsByFrequency)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return out
    
def anaVersionChanging():

    stemmer = nltk.stem.porter.PorterStemmer()
    stopword = set(nltk.corpus.stopwords.words('english'))
    f = codecs.open('../experiment/LDA-bugreport/keywords.txt',encoding='UTF-8')
    line = f.readline()
    while line:
        stopword.add(line[:-1])
        line = f.readline()
    f.close()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[A-Za-z]+')
    globalDict = nltk.FreqDist()
    f = codecs.open("reviews.txt", "r",encoding="utf-8")
    g = open("reviewindex.txt","w")
    dicindex = 0
    dic = {}
    data = []
    index = 0
    for l in f:
        if index % 10000 == 0:
            print "readreview, No.", index
            print len(dic.keys())
        index += 1
        info = eval(l)
        appid = info['aid']
        uid = info['uid']
        ver = info['version']
        rating = info['rating']
        text = info['title'] + info['text']
        text = tokenizer.tokenize(text)
        
        # stopwords, short words
        words = [token.lower() for token in text if token.lower() not in stopword and len(token) > 2]

        s = []
        for word in words:
            s.extend(infer_spaces(word))

        # again
        words = [token.lower() for token in s if token.lower() not in stopword and len(token) > 2]
        # stemming
        words = [stemmer.stem(ss) for ss in s]
        
        # word to num
        words2num = []
        for w in words:
            try:
                x = dic[w]
            except Exception, e:
                #print w
                dic[w] = dicindex
                dicindex += 1
        
        for w in words:
            words2num.append(dic[w])
        
        #data.append(words2num)
        
        print >> g, (appid, ver, words2num, uid, rating)
        
    f.close()
    g.close()
    
    f = open("words2num.txt","w")
    for x in dic:
        print >> f, (dic[x], x)
    f.close()
    '''   
    lda = LdaGibbsSampler(documents, 20)
    lda.configure(3000, 1000, 10)
    lda.gibbs()
    theta = lda.get_theta()
    
    f = open("theta.txt","w")
    for t in theta:
        print >> f, t
    f.close()
    ''' 
'''   
def delLow():
    wordTF = {}
    lowWord = {}
    f = open("reviewindex.txt","r")
    index = 1
    for l in f:
        if index % 10000 == 0:
            print "read step1, No,",index
        index += 1
        x = eval(l)
        for word in x[2]:
            wordTF[word] = 1 + wordTF.get(word, 0)
    f.close()
    
    lowNum = 100
    for word in wordTF:
        if wordTF[word] < lowNum:
            lowWord[word] = 1
    index = 1
    f = open("reviewindex.txt","r")
    g = open("reviewLDA.txt","w")
    for l in f:
        if index % 10000 == 0:
            print "read step2, No,",index
        index += 1
        x = eval(l)
        words = []
        for word in x[2]:
            try:
                z = lowWord[word]
                words.append(word)
            except Exception, e:
                pass
        print >> g, (x[0], x[1], words)
    g.close()
    f.close()
'''
''' 
def replaceABC(x):
    x = x.replace("a","")
    x = x.replace("b","")
    x = x.replace("c","")
    x = x.replace("d","")
    x = x.replace("e","")
    return x
    
def anaVersionRating():
    f = open("verreNum.txt","r")
    g = open("verRating.txt","w")
    index = 1
    appVR = []
    for l in f:
        appVR = []
        info = eval(l)
        if info[0] not in useful:
            continue
        ratings = info[1]["ratings"]
        for version in ratings:
            v = version.split('.')
            #v = version.replace("\W","")
            if len(v) > 2:
                v[2] = replaceABC(v[2])
            ver = int(v[0]) * 1000000
            if len(v) > 1:
                ver += int(v[1] * 1000)
            if len(v) > 2 and v[2].isdigit():
                ver += int(v[2])
            #v = [int[i] for i in v]
            r = ratings[version]
            ave = float(sum(r))/len(r)
            appVR.append((info[0],ave,ver,version,len(r)))
        appVR = sorted(appVR, key=lambda appVR: appVR[3])
        for line in appVR:
            print >> g, (line[0],line[1],line[3],line[4])
            
    f.close()
    g.close()
    
def anaTime():
    f = open("reviews.txt","r")
    g = open("verTime.txt","w")
    data = {}
    #index = 1
    for l in f:
        #print "review no.",index
        #index += 1
        info = eval(l)
        #print info
        #exit()
        aid = info['aid']
        #if aid not in useful:
            #continue
        try:
            x = data[aid]
        except Exception, e:
            data[aid] = {}
            
        ver = info["version"]
        t = int(info["time"])
        try:
            x = data[aid][ver]
        except Exception, e:
            data[aid][ver] = t
        
        if t < data[aid]:
            data[aid][ver] = t
    for id in data:
        app = data[id]
        li = sorted(app.items(), lambda x, y: cmp(x[1],y[1]))
        timein = []
        for x in xrange(len(li)-1):
            y = li[x+1][1] - li[x][1]
            #tstring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(y)))
            timein.append(y)
        xi = [li[0][0]]
        for y in xrange(len(timein)):
            xi.append(timein[y])
            xi.append(li[y+1])
        print >> g, (id, xi)
        #tstring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(t)))
        
    f.close()
    g.close()
'''
if __name__ == "__main__":
    anaVersionChanging()
