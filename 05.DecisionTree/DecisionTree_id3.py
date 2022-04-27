import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, axis=None, x=None, y=None, data=None, depth=None, parent=None):
        self.axis = axis  # 使用哪一个特征作为分类标准
        self.children = {}  # 子节点
        self.y = y  # 类标记，只有叶子节点有
        self.data = data  # 包含哪些数据，只有叶子节点有
        self.depth = depth
        self.parent = parent


class ID3:
    def __init__(self, epsilon=0, alpha=0):
        self.epsilon = epsilon  # 节点继续分裂的信息增益阈值
        self.alpha = alpha  # 平衡拟合情况和树复杂程度的参数
        self.tree = Node(depth=0)

    # 计算当前数据集的熵
    def cal_entropy(self, Y_train):
        probs = []
        counter = Counter(Y_train)
        for x, y in counter.most_common():
            probs.append(y / len(Y_train))
        entropy = -np.sum(np.multiply(probs, np.log2(probs)))  # 逐元素矩阵相乘
        return entropy

    def cal_conditional_entropy(self, dataset, col):
        cond_entropy = 0
        col_counter = Counter(dataset[col])
        for x, y in col_counter.items():
            cond_entropy += self.cal_entropy(dataset[dataset[col] == x]["label"]) * y / dataset.shape[0]
        return cond_entropy

    def build(self, dataset, node):
        X_train = dataset.iloc[:, 0:-1]
        Y_train = dataset["label"]
        # Case1: 所有实例属于同一类，返回单节点树
        if len(Counter(dataset["label"])) == 1:
            node.y = list(Counter(dataset["label"]).keys())[0]
            node.data = dataset
        # Case2: 特征为空，将数据集中实例数最大的类作为该节点的标记
        elif X_train is None:
            node.y = len(Counter(dataset["label"][0]).most_common(1)[0][0])
            node.data = dataset
        # Case3: 计算各特征的信息增益，选择信息增益最大的特征
        else:
            entropy = self.cal_entropy(dataset["label"])
            columns = X_train.columns
            gains = []
            for col in columns:
                info_gain = entropy - self.cal_conditional_entropy(dataset, col)
                gains.append(info_gain)
            max_gain = max(gains)
            max_gain_col = columns[gains.index(max_gain)]
            # print(columns)
            # print(max_gain, gains.index(max_gain), max_gain_col)
            # 判断最大信息增益是否大于设定阈值
            # 若大于等于，决策树向下生长；否则，返回一棵单节点树
            if max_gain >= self.epsilon:  # 决策树向下生长
                node.axis = max_gain_col
                col_counter = Counter(dataset[node.axis])
                for x in col_counter.keys():
                    child_dataset = dataset[dataset[node.axis] == x]
                    child_dataset = child_dataset.drop(axis=1, columns=node.axis)
                    child = Node(depth=node.depth + 1, parent=node)
                    self.build(child_dataset, child)
                    node.children[x] = child
            else:  # 返回一棵单节点树
                node.y = len(Counter(dataset["label"][0]).most_common(1)[0][0])
                node.data = dataset

            # for col in range(dataset.shape[1] - 1):
            #     feature = dataset[:, col]
            #     print(feature)

    def fit(self, dataset):
        self.build(dataset, self.tree)

    def showNode(self, node):
        if len(node.children) == 0:  # 到达叶子节点
            print("result: ", node.y)
            print(node.data)
        else:
            print("depth:{}, axis:{}".format(node.depth, node.axis))
            for k, v in node.children.items():
                print("next node: axis:{}, class:{}".format(node.axis, k))
                self.showNode(v)

    def show(self):
        self.showNode(self.tree)

    def predictNode(self, x, node):
        if len(node.children) == 0:  # 到达叶子节点，返回分类结果
            return node.y
        axis = node.axis
        res = self.predictNode(x, node.children[x[axis]])
        return res

    def predict(self, X_dftest):
        results = []
        for i in range(X_dftest.shape[0]):
            x = X_dftest.iloc[i, :]
            res = self.predictNode(x, self.tree)
            results.append(res)
        return results

    # 缺少剪枝部分
    def _prune(self, root, dataset):
        X = dataset.iloc[:, 0:-1]
        Y = dataset["label"]
        # 计算剪枝后的损失函数
        # 即现在的root就是叶子节点，不再向下生长
        pruned_entropy = len(X) * self.cal_entropy(Y)
        pruned_loss = pruned_entropy + self.alpha
        # root是叶子节点，直接返回
        if len(root.children) == 0:
            return pruned_loss
        cur_loss = 0.
        for col_val in root.children:
            child_dataset = dataset[dataset[root.axis] == col_val]
            child = root.children[col_val]
            cur_loss += self._prune(child, child_dataset)
        print("loss if prune: ", pruned_loss)
        print("current loss: ", cur_loss)
        if pruned_loss < cur_loss: # 若剪枝后的损失函数比现在的小，剪枝
            root.children.clear()
            return pruned_loss
        return cur_loss # 不剪枝

    def prune(self, dataset):
        self._prune(self.tree, dataset)


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

    id3 = ID3()
    id3.fit(dataset)
    # id3.show()

    X_test = [["中年", "否", "是", "一般"],
              ["青年", "是", "否", "一般"]]
    X_dftest = pd.DataFrame(X_test, columns=["年龄", "有工作", "有自己的房子", "信贷情况"])
    pred = id3.predict(X_dftest)
    print(pred)

    id3.prune(dataset)
