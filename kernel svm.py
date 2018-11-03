# _*_coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():         #逐行读取
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(float(lineArr[2]))                     #添加数据标签
    return dataMat,labelMat

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
"""函数说明：核转换函数"""
class optStruct:
    def __init__(self,dataMatin,classLabels,C,toler,kTup):
        self.X=dataMatin
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatin)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def kernelTrans(X,A,kTup):
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))
    else:raise NameError
    return K
def calcEk(oS, k):  # 辅助函数，计算E值，并返回
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 用于缓存误差
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)  # 加快计算速度
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek  # 选择具有最大步长的
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEK(oS, k):  # 计算误差值并存入缓存中
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
"""
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
def innerL(i, oS):
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H: print("L==H");return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i,j]- oS.K[i,i]-oS.K[j,j]
        if eta >= 0: print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEK(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough");
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEK(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] -\
             oS.labelMat[j] * (oS.alphas[j] - alphaJold)* oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i, :] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold)* oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else:oS.b = (b1 + b2) / 2.0
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
def smoP(dataMatin, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatin), np.mat(classLabels).transpose(), C, toler,kTup)   #初始化数据结构
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet,iter:%d i: %d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound,iter:%d i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number:%d" % iter)
    return oS.b,oS.alphas
"""函数说明：利用核函数进行分类的径向基测试函数"""
def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("there are %d support vectors" %np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):errorCount+=1
    print("the training error rate is :%f" %(float(errorCount)/m))
    dataArr,labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    m,n=np.shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount+=1
    print("the test error rate is :%f" %(float(errorCount)/m))
if __name__=='__main__':
    testRbf()
def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSetRBF.txt')                        #加载训练集
    showDataSet(dataArr, labelArr)
