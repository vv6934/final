from __future__ import absolute_import
import torch
import torch.nn as nn

# torch.cuda.set_device(0)
import numpy as np
import os

import ipdb
import matplotlib
from tqdm import tqdm
import torchvision
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, TestSimuDataset
from model import FasterRCNNVGG16

from model import FasterRCNNResNet50
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from utils.vis_tensorboard import writer
from trainer import LossTuple

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(
        enumerate(dataloader)
    ):
        if ii == 100:
            break
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        use_07_metric=True,
    )

    return result


def train(**kwargs):
    # 配置参数
    opt._parse(kwargs)

    # 加载数据
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # pin_memory=True,
        num_workers=opt.num_workers,
    )
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    test_simu_dataset = TestSimuDataset(opt)
    test_simu_dataloader = data_.DataLoader(
        test_simu_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    # 载入模型
    # faster_rcnn = FasterRCNNVGG16()
    faster_rcnn = FasterRCNNResNet50()

    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        train_epoch_loss = 0
        print('epoch:', epoch + 1)
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            train_epoch_loss += trainer.train_step(img, bbox, label, scale).total_loss

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(
                    ori_img_, at.tonumpy(bbox_[0]), at.tonumpy(label_[0])
                )
                trainer.vis.img('gt_img', gt_img)

                # plot predict bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
                    [ori_img_], visualize=True
                )
                pred_img = visdom_bbox(
                    ori_img_,
                    at.tonumpy(_bboxes[0]),
                    at.tonumpy(_labels[0]).reshape(-1),
                    at.tonumpy(_scores[0]),
                )
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img(
                    'roi_cm', at.totensor(trainer.roi_cm.conf, False).float()
                )

        # tensorboard train loss
        train_epoch_loss /= len(dataset)
        writer.add_scalar('train/loss', train_epoch_loss, epoch + 1)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        # tensorboard test acc
        writer.add_scalar('test/acc', np.mean(eval_result['ap']), epoch + 1)

        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(
            str(lr_), str(eval_result['map']), str(trainer.get_meter_data())
        )
        trainer.vis.log(log_info)

        # tensorboard test mAP
        writer.add_scalar('test/mAP', eval_result['map'], epoch + 1)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # tensorboard test loss
        test_epoch_loss = 0
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(test_simu_dataloader)):
            if ii == 100:
                break
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            with torch.no_grad():
                loss_tuple = trainer.forward(img, bbox, label, scale)
                test_epoch_loss += loss_tuple.total_loss

        test_epoch_loss /= 100
        writer.add_scalar('test/loss', test_epoch_loss, epoch + 1)

    trainer.save(save_path='./resnet_random_init.pth')


if __name__ == '__main__':
    # import fire

    # fire.Fire()
    train(env='fasterrcnn', plot_every=100)
