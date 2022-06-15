import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import numpy as np

logger = logging.getLogger(__name__)


def get_loader(local_rank, hp):
    # if local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    if hp.train.data_aug == 'normal':
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    elif hp.train.data_aug == 'cutmix':
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    elif hp.train.data_aug == 'mixup':
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    elif hp.train.data_aug == 'cutout':
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.RandomResizedCrop((hp.data.image_size, hp.data.image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    else:
        raise NameError('需在设置文件中(file-train-data_aug指定正确的数据增强模式，[normal, cutmix, mixup, cutout].')


    # transform_test = transforms.Compose([
    #     transforms.Resize((hp.data.image_size, hp.data.image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    transform_test = transforms.Compose([
        transforms.Resize((hp.data.image_size, hp.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    if hp.data.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=hp.data.path,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=hp.data.path,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root=hp.data.path,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=hp.data.path,
                                    train=False,
                                    download=True,
                                    transform=transform_test) if local_rank in [-1, 0] else None
    # if local_rank == 0:
    #     torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if local_rank == 0 else DistributedSampler(trainset)
    # test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              # sampler=train_sampler,
                              batch_size=hp.train.batch,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             # sampler=test_sampler,
                             batch_size=hp.train.valid_batch,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader