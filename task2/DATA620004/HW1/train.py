from transformers import BertForMaskedLM
from model import *
from data import DatasetLoader
import argparse
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
import pickle
import os

def pickle_dump(object, filename='save.pkl'):
    with open(os.path.join('./Results', filename), 'wb') as file:
        pickle.dump(object, file)


def pickle_read(filename='save.pkl'):
    with open(os.path.join('./Results', filename), 'rb') as file:
        data = pickle.load(file)
        return data



def train(
    X,
    Y,
    val_X,
    val_Y,
    data_loader,
    layers_dims,
    model,
    seed=1234,
    num_epochs=200,
    L2_lambda=0,
    learning_rate=0.01,
    decay_epoch=50,
    lr_decay=0.5,
    batch_size=128,
):

    L = len(layers_dims)
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    # tensorboard 可视化训练过程
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir = './Results/tensorboard/' + TIMESTAMP
    writer = SummaryWriter(log_dir)


    # 初始化参数
    model.initialize(layers_dims, seed)

    best_parameters = {}
    best_accuracy = 0
    for epoch in range(num_epochs):
        for step, (batch_X, batch_Y) in enumerate(data_loader.batch_generator(X, Y, batch_size, seed=seed+epoch)):
            # 1. 归一化
            batch_X = model.normalize(batch_X)
            # 2. 前向传播
            batch_Y_hat, cache = model.forward(batch_X)
            # 3. 计算误差
            loss = model.compute_loss(batch_Y_hat, batch_Y, L2_lambda)
            # 4. 反向传播
            grads = model.backward(batch_X, batch_Y, cache, L2_lambda)
            # 5. 更新参数
            model.update_parameters_with_sgd(grads, learning_rate)

        # 6. 计算整个训练集的误差和准确率
        accuracy, loss = model.compute_accuracy(X, Y, L2_lambda)
        train_accuracy.append(accuracy)
        train_loss.append(loss)

        writer.add_scalar("train_loss", loss, epoch)
        writer.add_scalar("train_accuracy", accuracy, epoch)


        print(f"Epoch: {epoch}, train set: loss {loss}, accuracy {accuracy}")

        # 7. 计算验证集的误差和准确率：选择在验证集上表现最优的模型参数
        accuracy, loss = model.compute_accuracy(val_X, val_Y, L2_lambda)
        val_accuracy.append(accuracy)
        val_loss.append(loss)

        writer.add_scalar("val_loss", loss, epoch)
        writer.add_scalar("val_accuracy", accuracy, epoch)
        writer.close()

        print(f"Epoch: {epoch}, val set: loss {loss}, accuracy {accuracy}")
        if accuracy > best_accuracy:
            best_parameters = model.parameters
            best_accuracy = accuracy

        if decay_epoch > 0 and (epoch + 1) % decay_epoch == 0:
            learning_rate = model.adjust_learning_rate(learning_rate, decay=lr_decay)

    object = {
        'parameter': best_parameters,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'best_accuracy': best_accuracy,
    }
    return object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./Datasets/MNIST")
    parser.add_argument("--rate", default=0.3, help="validation set ratio")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4396)
    parser.add_argument("--epoch", type=int, default=45)
    parser.add_argument("--L2_lambda", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument("--decay_epoch", type=int, default=20, help="learning rate decay epoch")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay rate")
    args = parser.parse_args()
    name = f"hd{args.hidden_dim}_reg{args.L2_lambda}_lr{args.lr}"
    data_loader = DatasetLoader()
    train_images, train_labels = data_loader.load_mnist(args.data_path)
    # 在验证集上选择表现最好的模型
    train_X, train_Y, val_X, val_Y = data_loader.split_train_val(
        train_images, train_labels, args.rate
    )
    # 784  --> hidden_dim --> 10
    layer_dims = [train_images.shape[1], args.hidden_dim, 10]
    model = TwoLayerNN()
    results = train(
        train_X,
        train_Y,
        val_X,
        val_Y,
        data_loader,
        layer_dims,
        model,
        seed = args.seed,
        num_epochs=args.epoch,
        L2_lambda=args.L2_lambda,
        learning_rate=args.lr,
        decay_epoch=args.decay_epoch,
        lr_decay=args.lr_decay,
        batch_size = args.batch_size
    )
    save_filename = 'model_{}.pkl'.format(name)
    pickle_dump(results, save_filename)

