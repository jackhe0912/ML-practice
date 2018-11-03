def loaddataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
"""Apriori算法中的辅助函数
   1：创建集合C1，C1是大小为1的所有候选集的集合"""
def createC1(dataset):
    C1=[]
    m=len(dataset)
    for i in range(m):
        for item in dataset[i]:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))
print(createC1(loaddataset()))
"""计算所有项集的支持度函数"""
def scanD(D,Ck,minSupport):
    ssCnt={}
    m=len(D)
    for i  in range(m):     #遍历数据集
        for can in Ck:      #遍历候选项集
            if can.issubset(D[i]):  #如果候选项集是数据集中其中一项集的子集
                if can not in ssCnt.keys():
                    ssCnt[can]=1
                else:ssCnt[can]+=1     #计算数据集中包含某项集的个数，并以字典的形式保存
    numItems=float(len(D))
    retlist=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems   #计算项集的支持度
        if support>=minSupport:      #如果项集的支持度大于等于设置的最小支持度，则存入
            retlist.insert(0,key)
        supportData[key]=support
    return retlist,supportData
D=list(map(set,loaddataset()))
C1=createC1(loaddataset())
L1,supportdata0=scanD(D,C1,0.5)
print(L1)
print(supportdata0)


