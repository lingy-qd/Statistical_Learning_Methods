import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, axis=None, x=None):
        self.axis=axis
        self.x=x
        self.y=None #当节点是叶子节点时，分类结果
        self.left=None
        self.right=None


class CARTClassifier:
    def __init__(self):
        self.tree=Node()

    def cal_gini(self, dataset, col):
        def gini(dataset):
            Y=dataset["label"]
            Y_counter = Counter(Y)
            c=len(dataset)
            g=1
            for y,yc in Y_counter.items():
                g = g-(yc/c)*(yc/c)
            return g
        xall = Counter(dataset[col])
        count=len(dataset)
        ginis=[]
        for x,xc in xall.items():
            g=0
            childData = dataset[dataset[col]==x]
            g+=gini(childData)*xc/count
            childDataNot = dataset[dataset[col]!=x]
            g+=gini(childDataNot)*(count-xc)/count
            ginis.append(g)
            if(len(xall)==2): # 一个特判，当一个特征只有两个可能取值的时候，实际上只有一个切分点
                break
        return min(ginis), list(xall.keys())[ginis.index(min(ginis))]

    def build(self, node, dataset):
        X = dataset.iloc[:, 0:-1]
        Y = dataset["label"]
        # Case1: 所有实例属于同一类，返回单节点树
        if len(Counter(dataset["label"])) == 1:
            node.y = list(Counter(dataset["label"]).keys())[0]
            node.data = dataset
        # Case2: 特征为空，将数据集中实例数最大的类作为该节点的标记
        elif X is None:
            node.y = len(Counter(dataset["label"][0]).most_common(1)[0][0])
            node.data = dataset
        # Case3: 树向下生长，对于所有可能的特征和切分点，计算gini指数，gini指数最小的为最优
        else:
            columns = X.columns
            mingini = 1 # gini指数一定小于1
            axis = None
            x = None
            for col in columns:
                col_mingini, col_x = self.cal_gini(dataset, col)
                if mingini>col_mingini: # 找到了基尼指数更小的特征及切分点
                    mingini, axis, x = col_mingini, col, col_x
            # 确定在当前节点的数据集下，最优的特征及切分点
            # 构造二叉树，递归建树
            node.axis=axis
            node.x=x
            childDataLeft=dataset[dataset[node.axis]==node.x]
            node.left=Node()
            self.build(node.left, childDataLeft)
            childDataRight=dataset[dataset[node.axis]!=node.x]
            node.right=Node()
            self.build(node.right, childDataRight)

    def fit(self, dataset):
        self.build(self.tree, dataset)

    def predictNode(self, node, x):
        if node.left is None: # 找到叶子节点
            return node.y
        if x[node.axis]==node.x: # 进入左子树
            res = self.predictNode(node.left, x)
        else:
            res = self.predictNode(node.right, x)
        return res

    def predict(self, X):
        results=[]
        for i in range(len(X)):
            x = X.iloc[i,:]
            results.append(self.predictNode(self.tree, x))
        return results



if __name__ == "__main__":
    X_train = [
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "否", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ]
    X_dftrain = pd.DataFrame(X_train, columns=["年龄", "有工作", "有自己的房子", "信贷情况"])
    Y_train = ["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"]
    Y_dftrain = pd.DataFrame(Y_train, columns=["label"])
    dataset = pd.concat([X_dftrain, Y_dftrain], axis=1)

    CART = CARTClassifier()
    CART.fit(dataset)

    X_test = [["中年", "否", "是", "一般"],
              ["青年", "是", "否", "一般"]]
    X_dftest = pd.DataFrame(X_test, columns=["年龄", "有工作", "有自己的房子", "信贷情况"])
    print(CART.predict(X_dftest))
