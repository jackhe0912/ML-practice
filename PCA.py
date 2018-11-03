from numpy import *
import numpy as np
from matplotlib import pyplot as plt
def loadDataSet(filename,delim='\t'):
    with open(filename) as fr:
        stringArr=[line.strip().split(delim) for line in fr.readlines()]
        datArr=[list(map(float,line)) for line in stringArr]
    return np.mat(datArr)
def pca(dataArr,topNfeat=999999):
    """
    20180924
    函数说明：pca计算函数
    :param dataArr: 数据集
    :param topNfeat: 可选参数：应用的N个特征
    :return: 返回降维后的数据和重构后返回的原始数据
    """
    meanVals=mean(dataArr,axis=0)  #求特征的均值
    meanRemoved=dataArr-meanVals   #去中心化，将坐标原点移到样本点的中心点
    covMat=cov(meanRemoved,rowvar=0)      #计算协方差矩阵
    eigVal,eigVects=np.linalg.eig(np.mat(covMat))  #计算特征值和特征向量
    eigValInd=argsort(eigVal)  #对所有特征值从小到大排序
    eigValInd=eigValInd[:-(topNfeat+1):-1]   #取所有特征值中N个最大的特征值
    regeigVects=eigVects[:,eigValInd]     #取N个特征值对应的特征向量
    lowDDataMat=meanRemoved*regeigVects   #将原始数据转换到新空间
    reconMat=(lowDDataMat*regeigVects.T)+meanVals   #重构后返回原始数据
    return eigVal,lowDDataMat,reconMat
if __name__=='__main__':
    dataArr=loadDataSet('ex2test.txt')
    plt.scatter(dataArr[:,0].tolist(),dataArr[:,1].tolist())
    plt.show()
    eigVal,lowData,reconMat=pca(dataArr,1)
    print(shape(dataArr))
    print(lowData)
    print(shape(lowData))
    print(eigVal)
"""函数说明：将NaN替换成平均值的函数"""
def replaceNanWithMean():
    dataMat=loadDataSet('','')
    numfeat=shape(dataMat)[1]
    for i in numfeat:
        meanVal=mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])   #求每个特征中非Nan的均值
        dataMat[nonzero(isnan(dataMat[:,i]))[0],i]=meanVal    #将每个特征中的Nan置为均值
    return dataMat





