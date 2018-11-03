#_*_ coding:utf-8 _*_
import numpy as np
from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],['maybe','not','take','him','to','dog','park','stupid']
                 ,['my','dalmation','is','so','cute','I','love','him'],['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]   # 1 代表侮辱性文字，0 代表正常言论
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet| set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)             #计算训练的文档数目
    numWords=len(trainMatrix[0])              #计算每篇文档的词条数
    pAbusive=sum(trainCategory)/float(numTrainDocs)    #文档属于侮辱类的概率
    p0Num=np.ones(numWords);p1Num=np.ones(numWords)  #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom=2.0; p1Denom=2.0                     #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i]==1:                 #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
            p1Num +=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:                                   #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
            p0Num +=trainMatrix[i]
            p0Denom +=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive               #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

if __name__=='__main__':
    listOposts,listClasses=loadDataSet()
    print(listOposts)
    myVocabList=createVocabList(listOposts)
    print(myVocabList)
    trainMat=[]
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    print(p0V)
    print(p1V)
    print(pAb)


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOposts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOposts)
    trainMat=[]
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
print(testingNB())
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
    return returnVec

"""函数说明：接收一个大字符串并将其解析为一个字符串列表"""

def textParse(bigString):              #将字符串转换为字符列表
    import re
    listOfTokens=re.split(r'\W*',bigString)       #将特殊符号作为切分标志进行字符切分，
    return [tok.lower() for tok in listOfTokens if len(tok)>2]   #除了单个字母，例如大写的I，其它单词变成小写

"""函数说明：测试贝叶斯分类器"""
def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):                      #遍历25个txt文件
        wordList=textParse(open('D:/machine learning/GitHub project/Machine-learning-master/NaiveBayes/email/spam/%d.txt' %i ).read())
        #读取字符串，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('D:/machine learning/GitHub project/Machine-learning-master/NaiveBayes/email/ham/%d.txt' %i ).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)                 #标记非垃圾邮件，1表示垃圾文件
    vocabList=createVocabList(docList)          #创建词汇表，不重复
    trainingSet=list(range(50));testSet=[]         #创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):                            #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex=int(random.uniform(0,len(trainingSet)))   #随机选取索索引值
        testSet.append(trainingSet[randIndex])              #添加测试集的索引值
        del(trainingSet[randIndex])                         #在训练集列表中删除添加到测试集的索引值
    trainMat=[];trainClasses=[]                                #创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:                               #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))    #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                    #将类别添加到训练集类别标签系向量中
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))    #训练朴素贝叶斯模型
    errorCount=0                                              #错误分类计数
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam)!=classList[docIndex]:  #如果分类错误
            errorCount+=1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%'% (float(errorCount) / len(testSet) * 100))
















