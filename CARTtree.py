"""CARTS算法代码实现
   20180911
   jackhe"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
"""读取数据函数"""
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename).readlines()
    for line in fr:
        curline=line.strip().split('\t')
        fltline=list(map(float,curline))
        dataMat.append(fltline)
    return dataMat
if __name__=='__main__':
    mydat=loadDataSet('ex00.txt')
    plt.scatter(np.mat(mydat)[:,0].A,np.mat(mydat)[:,1].A)
    plt.show()
    mydat = loadDataSet('ex0912.txt')
    plt.scatter(np.mat(mydat)[:, 1].A, np.mat(mydat)[:, 2].A)
    plt.show()
"""根据待切分特征和该特征的某个值对数据集进行切分"""
def binSplitDataSet(dataMat,feature,value):
    mat0=dataMat[np.nonzero(dataMat[:,feature]>value)[0],:]
    mat1=dataMat[np.nonzero(dataMat[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataMat):             #生成叶结点
    return mean(dataMat[:,-1])
def regErr(dataMat):              #计算目标变量的平方误差
    return var(dataMat[:,1])*shape(dataMat)[0]
def chooseBestSplit(dataMat,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0];tolN=ops[1]
    #如果不同剩余特征值的数目为1，则退出
    if len(set(dataMat[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataMat)
    m,n=np.shape(dataMat)
    S=errType(dataMat)
    bestS=inf;bestIndex=0;bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataMat[:,featIndex].T.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataMat,featIndex,splitVal)
            # 如果数据少于tolN,则退出
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #如果误差减小不大，则退出
    if (S-bestS)<tolS:
        return None,leafType(dataMat)
    #根据最佳的特征和最佳特征值切分数据集
    mat0,mat1=binSplitDataSet(dataMat,bestIndex,bestValue)
    #如果切分出的数据集很小，则退出
    if (shape(mat0)[0]<tolN)or (shape(mat1)[0]<tolN):
        return None,leafType(dataMat)
    return bestIndex,bestValue
"""树构建函数"""
def createTree(dataMat,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataMat,leafType,errType,ops)
    if feat==None:return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataMat,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
if __name__=='__main__':
    mydat=loadDataSet('ex00.txt')
    mydat=np.mat(mydat)
    t=createTree(mydat)
    print(t)
if __name__=='__main__':
    mydat=loadDataSet('ex0912.txt')
    mydat=np.mat(mydat)
    t=createTree(mydat)
    print(t)
"""函数说明：测试变量是否为一棵树
       parameter:
       odj - 输入变量
       return：
       返回布尔类型的结果
       jack 20180913"""
def isTree(obj):             #判断输入数据是否为一棵树
    return (type(obj).__name__=='dict')
def getMean(tree):                    #遍历树直到找到叶结点，并返回左结点与右结点的均值
    if isTree(tree['left']): tree['left']=getMean(tree['left'])
    if isTree(tree['right']): tree['right']=getMean(tree['right'])
    return (tree['left']+tree['right'])/2.0
"""函数说明：剪枝函数：
       parameter:
       tree - 待剪枝的树
       testData- 剪枝的测试集"""
def prune(tree,testData):
    if np.shape(testData)==0:    #测试集为空，则对树进行塌陷处理
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):  #左子树或者右子树存在，则切分测试数据集
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): tree['left']=prune(tree['left'],lSet)   #左子树存在，遍历左子树并剪枝
    if isTree(tree['right']): tree['right']=prune(tree['right'],rSet) #右子树存在，遍历右子树并剪枝
    if not isTree(tree['left']) and not isTree(tree['right']):      #如果当前左右结点为叶结点
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNomerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))  #计算没有合并的误差
        treeMean=(tree['left']+tree['right'])/2.0      #计算左结点和右结点的均值
        errorMerge=sum(power(testData[:,-1]-treeMean,2))  #计算合并后的误差
        if errorMerge<errorNomerge:   #如果合并后的 误差小于合并前的 误差
            print("merging")           #合并结点
            return treeMean            #返回合并结点均值
        else: return tree              #否则返回原树
    else: return tree                  #返回树
if __name__=='__main__':
    mydat = loadDataSet('ex2.txt')
    mydat = np.mat(mydat)
    mytree = createTree(mydat)
    print(mytree)
    testdata= loadDataSet('ex2test.txt')
    testdata=np.mat(testdata)
    cuttree=prune(mytree,testdata)
    print (cuttree)
def linearsolve(dataset):
    m,n=np.shape(dataset)
    X=np.mat(np.ones((m,n)));Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataset[:,0:n-1];Y=dataset[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('this matrix is singular,cannot do innverse,\n\ try increasing the second value od ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y
def modelLeaf(dataset):
    ws,X,Y=linearsolve(dataset)
    return ws
def modelErr(dataset):
    ws,X,Y=linearsolve(dataset)
    yHat=X*ws
    return sum(power(yHat-Y,2))
if __name__=='__main__':
    mydat = loadDataSet('exp2.txt')
    mydat = np.mat(mydat)
    plt.scatter(mydat[:,0].A,mydat[:,1].A)
    plt.show()
    mytree = createTree(mydat,modelLeaf,modelErr,(1,10))
    print(mytree)