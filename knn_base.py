import numpy as np
from collections import Counter


class KNN_Base:

    def __init__(self, X_train, Y_train, k=1, p=2):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = min(X_train.shape[0], k)
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
            knn = [knn_list[i][-1] for i in range(self.k)]
            res.append((Counter(knn).most_common()[0][0]))
        return np.array(res)


if __name__ == "__main__":

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.5, 0.8]])
    Y_train = np.array([1, 2, 3, 4, 5, 5])
    X = np.array([[0.3, 0.3], [0.8, 0.8]])

    knn_b = KNN_Base(X_train, Y_train, k=3, p=2)
    print("predict: ", knn_b.predict(X))
