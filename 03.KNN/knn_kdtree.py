import numpy as np
from collections import Counter


class KNN_Base:

    def __init__(self, X_train, Y_train, k=1, p=2):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = min(X_train.shape[0], k) # k-近邻，指定k值
        self.p = p  # 如何计算距离，p=1为曼哈顿距离，p=2为欧氏距离

    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            x = X[i]
            knn_list = []
            for i in range(self.X_train.shape[0]):  # 先计算前n个训练点和预测点的距离
                dist = np.linalg.norm(x - self.X_train[i], ord=self.p)
                knn_list.append([dist, self.Y_train[i]])
            knn_list.sort(key=lambda x: x[0], reverse=False)  # 按照dist升序排列
            #print(knn_list)
            knn = [knn_list[i][-1] for i in range(self.k)]
            res.append((Counter(knn).most_common()[0][0]))
        return np.array(res)


class KDNode:

    def __init__(self, data, label, axis, depth=0):
        self.left = None
        self.right = None
        self.parent = None
        self.axis = axis
        self.data = data
        self.label = label
        self.depth = depth


class kdTree:

    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.tree = self.build(X_train, Y_train, 0)

    def build(self, X, Y, depth=0):

        if not X.shape[0]:
            return None

        m, n = X.shape  # 训练数据集的数量和特征空间维度
        axis = depth % n  # 用于切分的坐标

        partition_idx = np.argpartition(X[:, axis], X.shape[0] // 2, axis=0)
        mid_idx = partition_idx[X.shape[0] // 2]
        split_point = float(X[mid_idx, axis])
        equal_x = X[mid_idx]
        equal_y = Y[mid_idx]
        tmp = X[:, axis] <= split_point
        tmp[mid_idx] = False
        less_x = X[tmp]
        less_y = Y[tmp]
        greater_x = X[X[:, axis] > split_point]
        greater_y = Y[X[:, axis] > split_point]

        node = KDNode(equal_x, equal_y, axis, depth)
        node.left = self.build(less_x, less_y, depth + 1)
        node.right = self.build(greater_x, greater_y, depth + 1)
        return node

    def preOrder(self, node):
        if node:
            print(node.data, node.label, node.axis)
            self.preOrder(node.left)
            self.preOrder(node.right)

    def query(self, X, k):

        n = X.shape[1]
        res = []
        for i in range(X.shape[0]):
            x = X[i]
            self.nearest = np.array([[-1, None] for j in range(k)]) # 开辟一个长度为k的数组

            def search(node):
                if node:
                    axis = node.depth % n
                    daxis = x[axis] - node.data[axis]
                    if daxis < 0:
                        search(node.left)
                    else:
                        search(node.right)
                    # 走到叶节点，计算x到叶子节点的距离dist
                    dist = np.linalg.norm(x - node.data, ord=2)
                    for j, d in enumerate(self.nearest): #对于当前节点，判断是否能插入到nearest数组中去
                        if d[0] < 0 or dist < d[0]:
                            self.nearest = np.insert(self.nearest, j, [dist, node], axis=0)
                            self.nearest = self.nearest[:-1]
                            break

                    # 开始回溯
                    temp = list(self.nearest[:,0]).count(-1) # 计算nearest数组中距离为-1的个数
                    # nearest[-temp-1, 0]是当前nearest中已有的最近点中，距离最大的点
                    # self.nearest[-n - 1, 0] > abs(daxis)代表以x为圆心，self.nearest[-n - 1, 0]为半径的圆与axis相交
                    # 说明在左右子树里可能有比self.nearest[-n-1, 0]更近的点，需要到另一边去递归
                    if self.nearest[-temp - 1, 0] > abs(daxis):
                        if daxis < 0: # 走的是搜索的另一边
                            search(node.right)
                        else:
                            search(node.left)

            search(self.tree)
            nodeList = self.nearest[:, 1]
            knn = [node.label for node in nodeList]
            res.append(Counter(knn).most_common()[0][0])
            print("x=", x, "\tpredict=", Counter(knn).most_common()[0][0], "\n")

        return np.array(res)


class knn_kdtree:

    def __init__(self, k=1, p=2):
        self.k = k
        self.p = p
        self.tree = None

    def fit(self, X_train, Y_train):
        self.tree = kdTree(X_train, Y_train)
        self.k = min(self.k, X_train.shape[0])

    def predict(self, X):
        topk = self.tree.query(X, self.k)
        return topk


if __name__ == "__main__":
    X_train = np.array([[2,3],[5,4],[4,7],[7,2],[8,1],[9,6]])
    Y_train = np.array([1, 2, 3, 4, 5, 6])
    X = np.array([[2.1,3.1],[2,4.5]])

    kdtree = kdTree(X_train, Y_train)
    #kdtree.preOrder(kdtree.tree)

    knn_kd = knn_kdtree(k=3, p=2)
    knn_kd.fit(X_train, Y_train)
    print("res kdtree:", knn_kd.predict(X))

    knn_b = KNN_Base(X_train, Y_train, k=3, p=2)
    print("res base:", knn_b.predict(X))
