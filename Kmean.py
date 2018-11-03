from numpy import *
import numpy as np
import math
import matplotlib.pyplot as plt
"""20180909
   Kmean 函数
   parameter：
   filename 文件名
   return:
   dataMat- 数据集"""
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        fltLine=list(map(float,curline))
        dataMat.append(fltLine)
    return dataMat
""" 函数说明：计算两个向量的欧式距离"""
def DistEclud(vecA,vecB):
    return math.sqrt(np.square(vecA-vecB).sum())
"""函数说明：给数据集构建一个包含k个随机质点的集合"""
def randCent(dataMat,k):
    n=np.shape(dataMat)[1]
    centriods=np.mat(np.zeros((k,n)))   #元素为0的质点矩阵
    for j in range(n):
        minJ=min(dataMat[:,j])
        rangeJ=float(max(dataMat[:,j])-minJ)
        centriods[:,j]=minJ+rangeJ*np.random.rand(k,1)
    return centriods
if __name__=='__main__':
    dataMat=np.mat(loadDataSet('kmeanstest.txt'))

def kMeans(dataMat,k,distMeas=DistEclud,createCent=randCent):
    m=np.shape(dataMat)[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    centoids=createCent(dataMat,k)
    clusterchanged=True
    while clusterchanged:
        clusterchanged=False
        for i in range(m):
            minDist=inf;minIndex=-1
            for j in range(k):
                distJI=distMeas(dataMat[i,:],centoids[j,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:clusterchanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centoids)
        for cent in range(k):
            ptsInClust=dataMat[nonzero(clusterAssment[:,0].A==cent)[0]]
            centoids[cent,:]=mean(ptsInClust,axis=0)
    return centoids,clusterAssment
if __name__=='__main__':
    dataMat=np.mat(loadDataSet('kmeanstest.txt'))
    plt.scatter(dataMat[:,0].A,dataMat[:,1].A)
    plt.show()
    centoid,clusters = kMeans(dataMat, k=3)
    print(clusters)




