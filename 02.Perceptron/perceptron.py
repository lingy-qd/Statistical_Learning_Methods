# 感知机是二分类的线性分类模型
#
# input：特征向量
# output：+1/-1
#
# 实现了感知机的原始算法


import numpy as np

class Perceptron:

    def __init__(self, lr=0.01, max_iteration=1000, verbose=True):
        self.lr = lr
        self.max_iteration = max_iteration
        self.verbose = verbose

    def train(self, X, Y):
        self.n = X.shape[1]  # 特征向量的维度
        self.w = np.random.rand(self.n)  # 权值
        self.b = np.random.rand(1)  # 偏移量

        epoch = 0
        while epoch < self.max_iteration:

            cnt = 0  # 记录这次epoch分错的数量
            lst = np.random.permutation(X.shape[0])
            for idx in lst:
                x, y = X[idx], Y[idx]
                if y * (self.w @ x + self.b) <= 0:  # 这里需要用到矩阵乘法
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    cnt += 1

            if (self.verbose):
                print("epoch={}, mis-classified={}".format(epoch, cnt))

            if cnt == 0:  # 全部分对，训练结束
                print("finish train: ")
                print("w: ", self.w)
                print("b: ", self.b)
                break

            epoch += 1

    def predict(self, X):
        Y = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):
            x = X[idx]
            if self.w @ x + self.b >= 0:
                Y[idx] = 1
            else:
                Y[idx] = -1
        return Y

    def evaluate(self, X, Y):
        predict = self.predict(X)
        correct = 0
        for idx in range(Y.shape[0]):
            if predict[idx] == Y[idx]:
                correct += 1
        print("accuracy:{}".format(correct / X.shape[0]))

    def get_parameters(self):
        return self.w, self.b

if __name__ == "__main__":

    x = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    x = np.array(x)
    y = np.array(y)

    perceptron = Perceptron(verbose=True)
    perceptron.train(x, y)
    perceptron.evaluate(x, y)
