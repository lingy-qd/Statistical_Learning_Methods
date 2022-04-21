import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import  LogisticRegression


class BinaryLogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, epsilon=0.001):
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def fit(self, X, Y):
        # 对训练数据进行扩充
        b = np.ones(X.shape[0])
        X = np.insert(X, X.shape[1], values=b, axis=1)
        self.feature_size = X.shape[1]
        self.w = np.random.rand(self.feature_size)

        for e in range(self.epochs):
            res = self.sigmoid(np.dot(self.w, X.transpose()))
            err = Y-res
            #err = np.sum(err)
            print(err)
            print(type(err))
            print(err.shape, self.w.shape, X.shape)
            self.w += self.lr*X.transpose()@err #注意这里梯度下降的写法，具体推导见笔记

    def predict(self, X):
        # 对训练数据进行扩充
        b = np.ones(X.shape[0])
        X = np.insert(X, X.shape[1], values=b, axis=1)
        res = self.sigmoid(np.dot(self.w, X.transpose()))
        res = (res>0.5)
        pred = [int(y) for y in res]
        return pred

    def score(self, X, Y):
        pred=self.predict(X)
        acc = (pred==Y).sum()/len(pred)
        return acc


if __name__ == "__main__":
    iris = load_iris()
    Xall = iris.data
    Yall = iris.target
    Yall = Yall.reshape(Yall.shape[0], 1)
    idx = [y == 0 or y == 1 for y in Yall]
    X = np.array([list(x) for i, x in zip(idx, Xall) if i])
    Y = np.array([y[0] for i, y in zip(idx, Yall) if i])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=33)
    # print(Counter(Y_train))
    # print(Counter(Y_test))

    clf = BinaryLogisticRegression(epochs=1000)
    clf.fit(X_train, Y_train)
    print("my accuracy: ", clf.score(X_test, Y_test))

    sklearn_clf = LogisticRegression(max_iter=1000)
    sklearn_clf.fit(X_train, Y_train)
    print("sklearn accuracy: ", sklearn_clf.score(X_test, Y_test))