import os

import matplotlib.pyplot as plt
import torch as t
from utils.config import opt
from model import FasterRCNNResNet50
# from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from train import eval
from data.dataset import TestDataset
from torch.utils import data as data_

# faster_rcnn = FasterRCNNVGG16()
faster_rcnn = FasterRCNNResNet50()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('./train.pth')
model_name = 'random_init'

opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model

# testset = TestDataset(opt)
# test_dataloader = data_.DataLoader(testset,
#                                    batch_size=1,
#                                    num_workers=opt.test_num_workers,
#                                    shuffle=False, \
#                                    pin_memory=True
#                                    )
# eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

# for i in range(1, 5):
#     path = f'./input/test_stage_one_{i}.jpg'
#     img = read_image(path)
#     img = t.from_numpy(img)[None]
#     trainer.faster_rcnn.show_rois(img, output_path=f'./output/test_stage_one_{i}.jpg', visualize=True)

for i in range(1, 4):
    path = f'./input/test_stage_two_{i}.jpg'
    img = read_image(path)
    img = t.from_numpy(img)[None]
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    vis_bbox(
        at.tonumpy(img[0]),
        at.tonumpy(_bboxes[0]),
        at.tonumpy(_labels[0]).reshape(-1),
        at.tonumpy(_scores[0]).reshape(-1),
        path=f'./output/{model_name}_{i}.jpg',
    )

# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it
