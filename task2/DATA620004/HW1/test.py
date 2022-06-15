import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from data import DatasetLoader
from train import pickle_read


if __name__ == '__main__':
    data_path = './Datasets/MNIST'
    best_model = ''
    file_path = 'Results'
    file_list = os.listdir(file_path)
    for file in file_list:
        if file.startswith('model_best'):
            best_model = file
            break
    data_loader = DatasetLoader()
    test_images, test_labels = data_loader.load_mnist(data_path, is_train=False)
    data = pickle_read(best_model)
    parameters = data['parameter']
    model = TwoLayerNN()
    model.parameters = parameters
    accuracy, loss = model.compute_accuracy(test_images, test_labels)
    print('Accuracy in test set is: {:.2%}'.format(accuracy))

