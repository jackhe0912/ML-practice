import numpy as np
import matplotlib.pyplot as plt
import math
"""函数说明：数据集"""
def loadSimpData():
    dataMat=np.matrix([[1.0,2.1],
              [2.0,1.1],
              [1.3,1.0],[1.0,1.0],[2.0,1.0]])
    classlabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classlabels
def showdataSet(dataMat,labelMat):
    xcord1=[];ycord1=[]
    xcord1_=[];ycord1_=[]
    n=len(labelMat)
    for i in range(n):
        if labelMat[i]==1.0:
            xcord1.append(dataMat[i,0]);ycord1.append(dataMat[i,1])
        else:
            xcord1_.append(dataMat[i,0]);ycord1_.append(dataMat[i,1])
    plt.scatter(xcord1,ycord1)
    plt.scatter(xcord1_,ycord1_)
    plt.show()
if __name__=='__main__':
    dataMat,labelMat=loadSimpData()
    showdataSet(dataMat, labelMat)

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):      #对特征分类
    """函数说明：单层决策树分类函数
       parameter:
          dataMatrix -数据集
          dimen-第dimen列
          threshVal - 阈值
          threshIneq  - 标志
          returns:
              retArray  - 分类结果"""
    retArray=np.ones((np.shape(dataMatrix)[0],1))   #初始化retArray为1
    if threshIneq=='lt':   #less than
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0   #如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0  #如果大于阈值,则赋值为-1
    return retArray

def buildStump(dataArr,classlabels,D):
    """函数描述：找到数据集上的最佳单层决策树
       parameter：
       dataArr - 数据集
       classlabels - 数据标签
       D - 样本权重
       return:
       bestStump - 最佳单层决策树信息
       minError - 最小误差
       bestClassEst - 最佳分类结果"""
    dataMatrix=np.mat(dataArr);labelMat=np.mat(classlabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0;bestStump={};bestClassEst=np.mat(np.zeros((m,1)))
    minError=float('inf')             #最小误差初始化为正无穷大
    for i in range(n):                      #遍历所有特征
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max();   #找到特征中最小的值和最大值
        stepsize=(rangeMax-rangeMin)/numSteps                             #计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:                                   #大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal=(rangeMin+float(j)*stepsize)                    #计算阈值
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)    #计算分类结果
                errArr=np.mat(np.ones((m,1)))                                #初始化误差矩阵
                errArr[predictedVals==labelMat]=0                       #分类正确的,赋值为0
                weightedError=D.T*errArr                                 #计算误差
                #print"split:dim %d,thresh %.2f,thresh inequal:%s,the weighted error is %.3f" %(i,threshVal,inequal,weighyedError)
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst
if __name__=='__main__':
    D=np.mat(np.ones((5,1))/5)
    dataMat,classlabels=loadSimpData()
    print(buildStump(dataMat,classlabels,D))


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """函数说明：基于单层决策树的AdaBoost训练过程
        parament:
        dataArr - 数据集
        classLabels - 数据标签
        numIt- 迭代次数
        return:
        weakClassArr - 弱分类器"""
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        alpha=float(0.5*math.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha                                    #存储弱学习算法权重
        weakClassArr.append(bestStump)                              #存储单层决策树
        print("classEst:",classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)  #计算e的指数项
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()                                               #根据样本权重公式，更新样本权重
        aggClassEst+=alpha*classEst                                  #计算类别估计累计值
        print ("aggClassEst:",aggClassEst.T)
        aggErrors=np.mat(np.ones((m,1)))
        aggErrors[np.sign(aggClassEst)==np.mat(classLabels).T]=0      #计算误差
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        if errorRate==0.0:break                                                                #误差为0，退出循环
    return weakClassArr
if __name__=='__main__':
    dataMat,classLabels =loadSimpData()
    classfierArray=adaBoostTrainDS(dataMat,classLabels,9)

def adaClassify(datatoclass,classfierArr):
    """函数说明：adaboost分类函数
       parameter:
       datatoclass - 待分类数组
       classfierArr - 弱分类器组成的数组"""
    dataMatrix=np.mat(datatoclass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classfierArr)):
        classEst= stumpClassify(dataMatrix,classfierArr[i]['dim'],classfierArr[i]['thresh'],classfierArr[i]['ineq'])
        aggClassEst += classfierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
if __name__=='__main__':
    dataArr,labelArr=loadSimpData()
    classifierArr=adaBoostTrainDS(dataArr,labelArr,30)
    adaClassify([0,0],classifierArr)
"""函数说明：自适应数据加载函数"""
def loadDataSet(filename):
    numfeat=len(open(filename).readline().split('\t'))
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curline=line.strip().split('\t')
        for i in range(numfeat-1):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return dataMat,labelMat
"""函数说明：ROC曲线的绘制及AUC计算函数
    parameter：
    preStrengths - 分类器的预测强度
    classlabels - 数据标签"""
def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)          #绘制光标的位置
    ySum=0.0               #用于计算AUC
    numPosClas=sum(np.array(classLabels)==1.0)  #统计数据类别中正例的个数
    yStep=1/float(numPosClas)                   #Y轴的步长
    xStep=1/float(len(classLabels)-numPosClas)             #X轴的步长
    sortedIndicies=predStrengths.argsort()      #预测强度从小到大排序
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:  #从排名最低的样例开始，所有排名更低的样例都判为反例，而排名更高的样例都判为正例
        if classLabels[index]==1.0:          #从[1,1]开始，样例属于正例，对真阳率进行修改；
            delX=0;delY=yStep;
        else:
            delX=xStep;delY=0;                #样例属于反例，对假阳率进行修改
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is ",ySum*xStep)








