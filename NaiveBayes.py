# 朴素贝叶斯算法+拉普拉斯平滑

import numpy as np
from collections import Counter, defaultdict


class NaiveBayes:

    def __init__(self, lamda=1, verbose=True):
        self.lamda = lamda
        self.verbose = verbose
        self.prior = {}  # 记录先验概率
        self.pa = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # 一个三维字典，记录条件概率分布，i-第i个类别，j-第j个特征，k-该特征可能取的第k个值

    def fit(self, X_train, Y_train):
        # 计算每个类的先验概率
        y_cnt = Counter(Y_train)
        for k, v in y_cnt.items():  # Counter类继承dict类，可以使用dict类中的方法
            self.prior[k] = (self.lamda + v) / (Y_train.shape[0] + self.lamda * len(y_cnt))

        # 计算条件概率分布
        for col in range(X_train.shape[1]):
            col_values = set(X_train[:, col])
            for x, y in zip(X_train, Y_train):
                self.pa[y][col][x[col]] += 1
            for y in y_cnt.keys():
                for a in self.pa[y][col]:
                    self.pa[y][col][a] += self.lamda
                    self.pa[y][col][a] /= (y_cnt[y] + self.lamda * len(col_values))

        if self.verbose:
            for y in self.prior:
                print(f"The prior probability of label {y} is", self.prior[y])
            for y in self.prior:
                for nth in self.pa[y]:
                    prob = self.pa[y][nth]
                    for a in prob:
                        print(f"When the label is {y}, the probability that {nth}th attribute be {a} is {prob[a]} ")

    def predict(self, X_test):
        labels = list(self.prior.keys())
        res = []
        for i in range(X_test.shape[0]):
            x = X_test[i]
            probs = []
            for y in labels:
                prob = self.prior[y]
                for j in range(len(x)):
                    prob *= self.pa[y][j][x[j]]
                probs.append(prob)
            if self.verbose:
                for y, p in zip(labels, probs):
                    print(f"The likelihood {x} belongs to {y} is {p}")
            res.append(labels[np.argmax(probs)])
        return np.array(res)


if __name__ == "__main__":
    X = np.array([[1, "S"],
                  [1, "M"],
                  [1, "M"],
                  [1, "S"],
                  [1, "S"],
                  [2, "S"],
                  [2, "M"],
                  [2, "M"],
                  [2, "L"],
                  [2, "L"],
                  [3, "L"],
                  [3, "M"],
                  [3, "M"],
                  [3, "L"],
                  [3, "L"]])
    Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    naive = NaiveBayes(lamda=1, verbose=True)
    naive.fit(X, Y)

    X_test = np.array([[2, "S"]])
    print(naive.predict(X_test))
