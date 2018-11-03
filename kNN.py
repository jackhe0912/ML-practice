#第一步导入数据
from numpy import *
import operator
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
#第二步k近邻算法
#inX用于分类的输入向量，dataSet训练样本集，标签向量labels,k表示用于选择最近邻的数据，必须是整数
def classify0(inX,dataSet,labels,k):
    #训练数据集的行数，这里因为dataSet后被赋值成group,所以，这里的dataSet.shape[0]=4L
    dataSetSize=dataSet.shape[0]
    #计算A，B之间的欧式距离
    #tile(A,B)表示对A重复B次，B可以是int型也可以是数组形式
    #**幂运算
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices=distances.argsort()
    # 定一个记录类别次数的字典
    classCount={}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel=labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]
if __name__=='__main__':
    group,labels=createDataSet()
    t=classify0([0,0],group,labels,3)
    print(t)
#将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    # 打开文件
    fr=open(filename)
    # 读取文件所有内容
    arrayOLines=fr.readlines()
    # 得到文件行数
    numberOfLines=len(arrayOLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat=zeros((numberOfLines,3))
    # 返回的分类标签向量
    classLabelVector=[]
    # 行的索引值
    index=0
    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line=line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine=line.split('\t')
        #将数据前三列提取出来, 存放到returnMat的NumPy矩阵中, 也就是特征矩阵
        returnMat[index,:]=listFromLine[0:3]
        #classLabelVector.append(int(listFromLine[-1])
        if listFromLine[-1]=='didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]=='largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
datingDataMat,datingLabels=file2matrix('DatingTestSet1.txt')

import matplotlib
import matplotlib.pyplot as plt
if __name__=='__main__':
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('DatingTestSet1.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is %d" % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount+=1.0
    print ("the total error rate is: %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList=['didntLike','smallDoses','largeDoses']
    percentTats=float(input("percentage of time spent playing video games"))
    iceCream=float(input("liters of ice cream consumed per year"))
    ffMiles=float(input("frequent flier miles earned per year"))
    datingDataMat,datingLabels=file2matrix('datingTestSet1.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("you will probably like this person:",resultList[classifierResult -1])
    






