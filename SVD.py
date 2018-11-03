"""20180925
   jackhe
   函数说明：奇异值分解"""
from numpy import *
U,sigma,VT=linalg.svd([[1,1],[7,7]])  #sigma矩阵只有对角线元素，且惯例是从大到小排列，这些对角元素称为奇异值
print(U)
print(sigma)
print(VT)
def loadExData():
    return[[1,1,1,0,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]
U,Sigma,VT=linalg.svd(loadExData())
print(Sigma)
print(U)
sig3=mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])  #构建sig3矩阵
print(sig3)
reConMat=U[:,:3]*sig3*VT[:3,: ]     #重构原始矩阵
print(reConMat)
from numpy import linalg as la
def euclidSim(inA,inB):     #欧氏距i离相似度
    return 1.0/(1.0+la.norm(inA-inB))  #当距离为0时，相似度为1，当 距离为无穷大时，相似度趋近为0
def pearsSim(inA,inB):            #皮尔逊相关系数相似度
    if len(inA)<3: return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]  #rowvar的值为bool值，该值为True或 1则行为变量，如果为False或者0，则列为变量
def cosSim(inA,inB):           #余弦相似度
    num=float(inA.T*inB)
    demon=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/demon)         #结果归一化0到1之间
mymat=mat(loadExData())
simil=euclidSim(mymat[:,0],mymat[:,0])
"""打印矩阵
    参数：inMat-输入矩阵
    thresh-阈值"""
def printmat(inMat,thresh=0.8):   #阈值可调，通过定义阈值来界定颜色深浅
    for i in range(32):          #遍历数据集的矩阵元素
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1)
            else:print(0)
        print ('')
"""图像压缩函数
    参数：numSV-选取的奇异值的个数
      thresh- 阈值"""
def imgCompress(numSV=3,thresh=0.8):
    myl=[]
    for line in open('').readlines():
        newrow=[]
        for i in range(32):
            newrow.append(int(line[i]))
        myl.append(newrow)
    mymat=mat(myl)
    printmat(mymat,thresh)   #打印矩阵
    U,Sigma,VT=la.svd(mymat)   #进行SVD分解
    Sigrecon=mat(zeros((numSV,numSV)))  #建立全0矩阵 ，用于创建奇异值对角矩阵
    for k in range(numSV):
        Sigrecon[k,k]=Sigma[k]            #创建奇异值矩阵
    reConMat=U[:,:numSV]*Sigrecon*VT[:numSV,:]    #重构奇异值分解后的矩阵
    printmat(reConMat,thresh)






