# from model import *
import numpy as np
import struct
import os


class DatasetLoader():

    def __init__(self):
        super(DatasetLoader, self).__init__()

    def load_mnist(self, path, is_train=True):
        """
        Load MNIST dataset
        """
        if is_train:
            labels_path = os.path.join(path, "train-labels.idx1-ubyte")
            images_path = os.path.join(path, "train-images.idx3-ubyte")
        else:
            labels_path = os.path.join(path, "t10k-labels.idx1-ubyte")
            images_path = os.path.join(path, "t10k-images.idx3-ubyte")
        with open(labels_path, "rb") as lb_path:
            magic, n = struct.unpack('>II', lb_path.read(8))
            labels = np.fromfile(lb_path, dtype=np.uint8)
        with open(images_path, "rb") as img_path:
            # [60000, 28, 28] --> [60000, 784]
            magic, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
            images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 784)
        # one-hot encoding
        labels = np.eye(10)[labels.reshape(-1)]
        return images, labels

    def split_train_val(self, X, Y, rate, seed=1234):
        """
        划分数据集
        """
        data_size = X.shape[0]
        val_size = int(data_size * rate)
        np.random.seed(seed)
        permutation = list(np.random.permutation(data_size))
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        train_X = X[train_indices]
        train_Y = Y[train_indices]
        val_X = X[val_indices]
        val_Y = Y[val_indices]
        return train_X, train_Y, val_X, val_Y

    def batch_generator(self, X, Y, batch_size=64, seed=1234):
        """
        batch 生成器

        参数：
            X：输入数据，维度为(样本的数量, 特征维度)
            Y：对应的是 X 的标签
            batch_size：每个 batch 的样本数量
            seed: 随机种子

        返回：
            一个 batch 的数据：(batch_X, batch_Y)
        """
        # 指定随机种子
        np.random.seed(seed)
        # 获取样本数量
        data_size = X.shape[0]
        for batch_count in range(data_size // batch_size):
            permutation = list(np.random.permutation(data_size))
            start = batch_count * batch_size
            end = start + batch_size
            batch_X = X[permutation[start:end], :]
            batch_Y = Y[permutation[start:end], :]
            mini_batch = (batch_X, batch_Y)
            yield mini_batch