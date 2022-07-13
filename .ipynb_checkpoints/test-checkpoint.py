from __future__ import print_function, absolute_import

import argparse
import torch
import torch.nn as nn
from tensorboard.backend.event_processing import event_accumulator as ea

torch.backends.cudnn.benchmark = True

from scripts.utils.misc import save_checkpoint, adjust_learning_rate
from scripts.utils.imgUtils import load_image, to_torch
from torchsummary import summary
import scripts.datasets as datasets
import scripts.machine as machines
from options import Options
import csv
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scripts.models.blocks import SEBlock
from scripts.models.split_cappV2 import *
from scripts.models.split_cappV3 import *


def main(args):
    val_loader = torch.utils.data.DataLoader(datasets.COCO('val', args), batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    data_loaders = (None, val_loader)
    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)
    Machine.test()


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        x.append(int(row[1]))
        y.append(float(row[2]))
    return x, y


def printlist(list=[]):
    list.append(1)
    return list


if __name__ == '__main__':
    # parser = Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    # main(parser.parse_args())
    # events = ea.EventAccumulator(r'/home/ipprlab/projects/ckpt/events.out.tfevents.1611904734.cisgsvr02')
    # events.Reload()
    # print(events.scalars.Keys())
    #
    # tensor1 = torch.tensor([[0.0, 2.1, 3, 4], [1, 2.1, 3, 4], [2, 2.1, 3, 4], [3, 2.1, 3, 4], [3, 2.1, 3, 4]])
    # masked = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
    # img = Image.open('/home/ipprlab/projects/Hday2night/masks/d64-131_1_1.jpg').convert("L")
    # img.show()
    # img = np.array(img)
    # print(str(img))

    name = 'f_1_1.jpg'
    new = name.split('_')
    print('_' +new[-2] +'_' + new[-1])
    # str = '012'
    # print(len(str))
    # tensor_reshape = tensor1.flatten(0, 1)
    # print(tensor1.shape)
    # tensor2 = torch.cat((tensor1, tensor1), dim=1)
    # print(tensor2.shape)
    # print(tensor1.flatten(0,1).shape)
    # nonzero = tensor_reshape.nonzero(as_tuple =False).flatten(0, 1)
    # nonzero = list(nonzero)
    # print(nonzero.pop(2))

    # model = SplitwithCAPPV3(shared_depth=3, blocks=5, long_skip=True).to('cuda')
    # # summary(encoder, input_size=(3, 256, 256))
    # summary(model, input_size=(512, 16, 16))
    # # list = nn.ModuleList()

    # que = [16, 32, 64, 128]
    # que1 = [17, 32, 64, 128]
    # que = que[:-1]
    # for id, item in enumerate(que):
    #     print(que1[id], item)
    # print(que)
    # for channel in [16, 32, 64, 128]:
    #
    #     list.append(SEBlock(channel=channel))
    # print(len(que))
    # for att, item in enumerate(list, 0):
    #     print(att, item)

    # plt.figure()
    #
    # x1, y1 = readcsv('/home/ipprlab/Downloads/run_.-tag-train_loss_TOTAL.csv')
    # plt.plot(x1, y1, label='loss_patch_bs2')
    #
    # x2, y2 = readcsv('/home/ipprlab/Downloads/run-.-tag-train_loss_TOTAL.csv')
    # plt.plot(x2, y2, label='loss_capp_bs4')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.grid()
    #
    # plt.show()

