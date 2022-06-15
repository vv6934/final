import numpy as np

class TwoLayerNN():

    def __init__(self):
        """
        parameters：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
        """
        super(TwoLayerNN, self).__init__()
        self.parameters = {}

    # 初始化网络参数
    def initialize(self, layer_dims, seed=1234):
        np.random.seed(seed)
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.normal(
                0.0, np.sqrt(2 / layer_dims[l - 1]), (layer_dims[l - 1], layer_dims[l])
            )
            self.parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))

    # 前向传播
    def forward(self, X):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        z1 = np.dot(X, W1) + b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.softmax(z2)
        cache = (z1, a1, W1, b1, z2, a2, W2, b2)
        return a2, cache

    # 反向传播
    def backward(self, X, Y, cache, L2_lambda):
        """
        两层网络的反向传播，L2 正则化

        输入：
        X：输入数据，维度为(输入节点数量，样本的数量)
        Y：X 对应的标签 (one-hot)
        cache: 前向传播中传递的值

        返回:
        gradients: 包含参数对应梯度的字典
        """
        n = X.shape[0]
        (z1, a1, W1, b1, z2, a2, W2, b2) = cache

        dz2 = 1.0 / n * (a2 - Y)
        dW2 = np.dot(a1.T, dz2) + (L2_lambda * W2) / n
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(X.T, dz1) + (L2_lambda * W1) / n
        db1 = np.sum(dz1, axis=0, keepdims=True)

        gradients = {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}

        return gradients


    def compute_loss(self, a2, Y, L2_lambda):
        """
        计算 loss
        """
        n = Y.shape[0]
        epsilon = 1e-8
        # np.multiply: 矩阵对应位置相乘
        cross_entropy = np.multiply(-np.log(a2 + epsilon), Y)
        loss = (1.0 / n) * np.sum(cross_entropy)
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        loss = loss + (np.sum(np.square(W1)) + np.sum(np.square(W2))) * L2_lambda / (2 * n)

        return loss

    # 参数更新函数
    def update_parameters_with_sgd(self, grads, learning_rate):

        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l + 1)] = (
                self.parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            self.parameters["b" + str(l + 1)] = (
                self.parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )



    # 学习率调整
    def adjust_learning_rate(self, lr, decay=0.1):
        return lr * decay


    # 计算准确率
    def compute_accuracy(self, test_X, test_Y, L2_lambda=0):
        # 归一化
        test_X = self.normalize(test_X)
        test_Y_hat, cache = self.forward(test_X)
        loss = self.compute_loss(test_Y_hat, test_Y, L2_lambda)
        pred_Y = np.argmax(test_Y_hat, axis=1)
        test_Y = np.argmax(test_Y, axis=1)
        accuracy = np.mean(test_Y == pred_Y)
        return accuracy, loss

     # 归一化函数
    def normalize(self, X):
        X = X.astype('float64')
        return X / np.max(X)

    # 激活函数
    def sigmoid(self, x):
        x = 1 / (1 + np.exp(-x))
        return x

    def relu(self, x):
        x = np.maximum(0, x)
        return x

    def softmax(self, x):
        M = np.max(x, axis=1, keepdims=True)
        x = x - M
        epsilon = 1e-8
        x = np.exp(x) / (epsilon + np.sum(np.exp(x), axis=1, keepdims=True))
        return x





