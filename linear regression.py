"""函数说明：回归函数与导入数据函数
    日期20180905"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(filename):
    numfeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curline=line.strip().split('\t')
        for i in range(numfeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return dataMat,labelMat
"""函数说明：回归函数"""
def standRegres(xArr,yArr):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws
if __name__=='__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    print(ws)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    yHat=xMat*ws
    t = corrcoef(yHat.T, yMat)  #求相关系数
    print(t)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
"""函数说明：局部加权线性回归函数
    parameter:
    testPoint - 测试数据集
    xArr - x轴数据集
    yArr - y轴数据集
    return:
    testPoint*ws -测试点的回归值"""
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    m=shape(xMat)[0]
    weights=np.mat(eye((m)))    #创建对角矩阵
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))  #权重值大小以指数级衰减
    xTx=xMat.T*(weights*xMat)
    if np.linalg.det(xTx)==0:
        print("this matrix is sinngular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws
"""用于数据集中每个点都调用lwlr函数，用于测试k大小"""
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
if __name__=='__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    print(yArr[0])
    t = lwlr(xArr[0], xArr, yArr, 1.0)
    print(t)
    yHat=lwlrTest(xArr,xArr,yArr,0.01)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=2,c='red')
    plt.show()
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()
if __name__=='__main__':
    abX,abY=loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat02 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat03 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print(rssError(abY[0:99],yHat01[0:99].T))
    print(rssError(abY[0:99],yHat02[0:99].T))
    print(rssError(abY[0:99],yHat03[0:99].T))
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print(rssError(abY[100:199],yHat01[0:99].T))
    yHat2 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print(rssError(abY[100:199], yHat2[0:99].T))
    yHat3 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print(rssError(abY[100:199], yHat2[0:99].T))
def ridgeregres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    demon=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(demon)==0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws=demon.I*(xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)            #mean() - 取算术平均值
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)                 #var() - 方差
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeregres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat
if __name__=='__main__':
    xArr,yArr=loadDataSet('abalone.txt')
    ridgeweights=ridgeTest(xArr,yArr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeweights)
    plt.show()
"""向前逐步回归"""
def stagewise(xArr,yArr,eps=0.01,numIt=100):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    yMean=mean(yMat,0)            #数据标准化
    yMat=yMat-yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)          # var() - 方差
    xMat = (xMat - xMeans)/xVar
    m,n=np.shape(xMat)
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1));wsTest=ws.copy();wsMax=ws.copy()
    for i in range(numIt):  #每一轮迭代中，
        print(ws.T)
        lowestError=inf      #设置最小误差为正无穷
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
if __name__=='__main__':
    xArr,yArr = loadDataSet('abalone.txt')
    t=stagewise(xArr,yArr,0.01,200)
    print(t)
