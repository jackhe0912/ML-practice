#_*_ coding:utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
"""函数说明：读取数据
   parameter：
   fileName- 文件名
   return:
   dataMat - 数据矩阵
   labelMat - 数据标签
   """
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():         #逐行读取
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(float(lineArr[2]))                     #添加数据标签
    return dataMat,labelMat
"""函数说明：数据可视化
   parameter：
   dataMat - 数据矩阵
   labelMat - 数据标签
   return:
   无
   """
def showDataSet(dataMat,labelMat):
    data_plus=[]
    data_minus=[]
    for i in range(len(dataMat)):
        if labelMat[i] ==1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)
    data_minus_np=np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1])
    plt.show()
if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat,labelMat)
"""函数说明:随机选择alpha
   parameter:
   i:alpha
   m:alpha参数个数
   return:
   j 
   """
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j
"""函数说明：修剪alpha
   parameter:
   aj - alpha
   H -  alpha上限
   L - alpha下限
   return：
   aj- alpha值
   """
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
"""函数说明：简化版SMO算法
    parameter:
    dataMatin - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
    return:
    alphas值"""
def smoSimple(dataMatin,classLabels,C,toler,maxIter):
    #转换成numpy的Matrix存储
    dataMatrix=np.mat(dataMatin);labelMat=np.mat(classLabels).transpose()
    #b值初始化为0，统计dataMatrix的维度
    b=0;m,n=np.shape(dataMatrix)
    alphas=np.mat(np.zeros((m,1)))    #初始化alpha参数，设为0
    iter=0        #初始化迭代次数
    while (iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            #步骤一：计算误差Ei
            fXi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])
            #优化alpha,设定一定的容错率
            if ((labelMat[i]*Ei< -toler)and (alphas[i]<C)) or ((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                j = selectJrand(i, m)          # 随机选择另一个与alpha_i成对优化的alpha_j
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b   #步骤一：计算误差Ej
                Ej = fXj - float(labelMat[j])
                alphaIold=alphas[i].copy();alphaJold=alphas[j].copy()                             # 保存更新前的aplpha值，使用深拷贝
                if (labelMat[i]!=labelMat[j]):             #步骤2：计算上下界L和H
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H: print("L=H");continue
                #步骤3：计算eta
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0");continue
                # 步骤4：更新alpha[j]
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                # 步骤5：修剪alpha_j
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough");continue
                # 步骤6：更新alpha_i
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                # 步骤7：更新b_1和b_2
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T\
                -labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T\
                -labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 步骤8：根据b_1和b_2更新b
                if (0<alphas[i]) and (C>alphas[i]):b=b1
                elif (0<alphas[j])and (C>alphas[j]):b=b2
                else: b=(b1+b2)/2.0
                # 统计优化次数
                alphaPairsChanged+=1
                # 打印统计信息
                print("iter:%d i :%d,pairs changed %d" %(iter,i,alphaPairsChanged))
        # 更新迭代次数
        if (alphaPairsChanged==0):iter+=1
        else:iter=0
        print("iteration number:%d" %iter)
    return b,alphas
if __name__=='__main__':
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alphas= smoSimple(dataArr,labelArr,0.6,0.001,40)
    print(b)
    print(alphas[alphas>0])
    print(np.shape(alphas[alphas>0]))
    for i in range(100):
        if alphas[i]>0.0:
            print(dataArr[i],labelArr[i])
"""函数说明：完整的Platt SMO算法加速优化"""
class optStruct:               #清理代码数据结构的函数
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))   #缓存误差函数 对Ei进行缓存的函数

def calcEk(oS,k):         #辅助函数，计算E值，并返回
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]    #用于缓存误差
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k ==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)   #加快计算速度
            if (deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek  #选择具有最大步长的
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
def updateEK(oS,k):    #计算误差值并存入缓存中
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if ((oS.labelMat[i]*Ei< -oS.tol)and(oS.alphas[i]<oS.C))or ((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        if (oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[i]+oS.alphas[j]-oS.C)
            H=min(oS.C,oS.alphas[i]+oS.alphas[j])
        if L==H: print("L==H");return 0
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0: print("eta>=0");return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEK(oS,j)
        if (abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough");return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEK(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)\
        *oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i,:]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)\
        *oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i])and (oS.C>oS.alphas[i]):oS.b=b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0
        return 1
    else:return 0
"""函数说明：完整版platt SMO的循环
    20180905
    parameter:
    dataMatin-数据集
    classLabels- 数据类别
    C  - 松弛变量
    toler -
    maxIter - 最大迭代次数
    kTup"""
def smoP(dataMatin,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(np.mat(dataMatin),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True;alphaPairsChanged=0
    while (iter<maxIter)and((alphaPairsChanged>0)or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("fullSet,iter:%d i: %d,pairs changed %d" %(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print ("non-bound,iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged) )
            iter+=1
        if entireSet:entireSet=False
        elif (alphaPairsChanged==0):entireSet=True
        print("iteration number:%d" %iter)
    return oS.b,oS.alphas
if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
    print(alphas[alphas>0])
    for i in range(len(dataArr)):
        if alphas[i]>0.0:
            print(dataArr[i],labelArr[i])
#
def calcWs(alphas,dataArr,labelArr):
    X=np.mat(dataArr);labelMat=np.mat(labelArr).T
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
    w=calcWs(alphas,dataArr,labelArr)
    print(w)
    dataMat=np.mat(dataArr)
    print(dataMat[0]*np.mat(w)+b)
    print(labelArr[0])
def showClassifer(dataMat, classLabels, w, b):
    """
    分类结果可视化
    Parameters:
        dataMat - 数据矩阵
        w - 直线法向量
        b - 直线解决
    Returns:
        无
    """
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()
if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
    w = calcWs(alphas,dataArr, classLabels)
    showClassifer(dataArr, classLabels, w, b)


