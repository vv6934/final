from model import *
from train import *
from data import DatasetLoader
import os

data_path = './Datasets/MNIST'
result_path = './Results'
seed = 1997
batch_size = 128
num_epochs = 80
decay_epoch = 50
lr_decay = 0.5
L2_lambdas = [0, 1e-1, 1e-2, 1e-3, 1e-4]
hidden_dims = [800, 500,300,100,50]
lrs = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]


# 在验证集上进行超参数的选择
if __name__ == '__main__':
    if os.path.exists(result_path) == 0:
        os.makedirs(result_path)
    data_loader = DatasetLoader()
    train_images, train_labels = data_loader.load_mnist(data_path)
    train_X, train_Y, val_X, val_Y = data_loader.split_train_val(train_images, train_labels, 0.3)

    best_model = {}
    best_name = ''
    best_accuracy = 0
    model = TwoLayerNN()
    for hidden_dim in hidden_dims:
        for L2_lambda in L2_lambdas:
            for lr in lrs:
                name = 'hd{}_reg{}_lr{}'.format(hidden_dim, L2_lambda, lr)
                layer_dims = [train_images.shape[1], hidden_dim, 10]
                results = train(
                    train_X,
                    train_Y,
                    val_X,
                    val_Y,
                    data_loader,
                    layer_dims,
                    model,
                    seed = seed,
                    num_epochs=num_epochs,
                    L2_lambda=L2_lambda,
                    learning_rate=lr,
                    decay_epoch=decay_epoch,
                    lr_decay=lr_decay,
                    batch_size=batch_size
                )
                accuracy = results['best_accuracy']
                save_filename = 'model_{}.pkl'.format(name)
                pickle_dump(results, save_filename)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = results
                    best_name = name


    save_filename = 'model_best_{}.pkl'.format(best_name)
    pickle_dump(best_model, save_filename)

