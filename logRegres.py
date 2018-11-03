import math
import numpy as np
import matplotlib.pyplot as plt
"""函数说明：加载数据"""
def loadDataSet():
    dataMat=[];labelMat=[]
    with open('D:/machine learning/GitHub project/Machine-learning-master/Logistic/testSet.txt') as fr:
        for line in fr.readlines():
            lineArr=line.strip().split('\t')
            dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    fr.close
    return dataMat,labelMat
"""sigmoid函数"""
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
"""函数说明：梯度下降算法"""
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)            #转换成numpy的mat
    labelMat=np.mat(classLabels).transpose()   #转换成numpy的mat,并进行转置
    m,n=np.shape(dataMatrix)                  #dataMatrix 的行数和列数
    alpha=0.001                                #移动步长,也就是学习速率,控制更新的幅度
    maxCycles=500                             #最大迭代次数
    weight=np.ones((n,1))                     #初始循环向量值
    for i in range(maxCycles):
        h=sigmoid(dataMatrix*weight)          #梯度上升矢量化公式
        error=(labelMat-h)
        weight=weight+alpha*dataMatrix.transpose()*error
    return weight
if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    print(gradAscent(dataMat,labelMat))
"""画出散点图和最佳拟合函数"""
def plotbestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xt1=[]
    xf1=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xt1.append(dataArr[i])
        else:
            xf1.append(dataArr[i])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(np.array(xt1)[:,1],np.array(xt1)[:,2],s=30,c='red',marker='s')
    ax.scatter(np.array(xf1)[:,1],np.array(xf1)[:,2],s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()
"""随机梯度下降法"""
def stocgradAscent0(dataMatIn,classLabels):
    m,n=np.shape(dataMatIn)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatIn[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatIn[i]
    return weights
def stocgradAscent1(dataMatIn,classLabels,numIter=500):
    """改进的随机梯度算法"""
    m,n=np.shape(dataMatIn)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha = 4/(1+j+i)+0.01           #降低alpha的大小
            randindex=int(np.random.uniform(0,len(dataIndex)))   #随机选取样本
            h = sigmoid(sum(dataMatIn[randindex] * weights))    #随机选取一个样本，计算h
            error = classLabels[randindex] - h                   #计算误差
            weights = weights + alpha * error * dataMatIn[randindex]   #更新回归系数
            del(dataIndex[randindex])                              #删除已经使用的样本
    return weights
if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    print(stocgradAscent1(np.array(dataMat),labelMat,numIter=500))
    weights = stocgradAscent1(np.array(dataMat), labelMat, numIter=500)
    print(plotbestFit(weights))
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    """用logistic回归进行分类"""
    frtrain=open('horseColicTraining.txt')
    frtest=open('horseColicTest.txt')
    frtrainlines=frtrain.readlines()
    frtestlines=frtest.readlines()
    frtrain.close()
    frtest.close()
    trainingSet=[];trainingLabels=[]
    for line in frtrainlines:       #读取数据集和标签
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights=stocgradAscent1(np.array(trainingSet), trainingLabels,numIter=500)
    errorCount=0;numTestVec=0.0
    for line in frtestlines:
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()     #调用colicTest()函数，并计算10次的总误差
    print("after %d iteration the average error rate is: %f" %(numTests,errorSum/float(numTests)))
if __name__=='__main__':
    print(colicTest())
    print(multiTest())




