from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imgUtils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCO(data.Dataset):
    def __init__(self, mode=None, config=None, sample=[], gan_norm=False):
        self.train = []
        self.anno = []
        self.mask = []

        self.input_size = config.input_size
        self.base_folder = config.base_dir
        self.dataset = config.data
        self.mode = mode

        if config == None:
            self.data_augumentation = False
        else:
            self.data_augumentation = config.data_augumentation

        self.isTrain = False if self.dataset.find('train') == -1 else True
        print(self.isTrain)

        mypath = join(self.base_folder, self.dataset)
        if self.mode == 'train':
            file_names = sorted([f for f in listdir(join(mypath,'train', 'composite_images')) if isfile(join(mypath,'train', 'composite_images', f))])
        elif self.mode == 'test':
            file_names = sorted([f for f in listdir(join(mypath,'test', 'composite_images')) if isfile(join(mypath,'test', 'composite_images', f))])
        elif self.mode == 'val':
            file_names = sorted([f for f in listdir(join(mypath,'test', 'composite_images')) if isfile(join(mypath,'test', 'composite_images', f))])

        if config.limited_dataset > 0:
            x_train = sorted(list(set([ file_name.split('_')[0] for file_name in file_names ])))
            tmp = []
            for x in x_train[:config.limited_dataset]:
                #get the file_name by identifier
                tmp.append([y for y in file_names if x in y][0])
            file_names = tmp
        else:
            file_names = file_names

        if self.mode == 'train':
            for file_name in file_names:
                self.train.append(os.path.join(mypath, self.mode, 'composite_images', file_name))
                self.mask.append(os.path.join(mypath, self.mode, 'masks', file_name))
                self.anno.append(os.path.join(mypath, self.mode,'real_images', file_name))
        elif self.mode == 'test':
            for file_name in file_names:
                self.train.append(os.path.join(mypath, self.mode, 'composite_images', file_name))
                self.mask.append(os.path.join(mypath, self.mode, 'masks', file_name))
                self.anno.append(os.path.join(mypath, self.mode,'real_images', file_name))
        elif self.mode == 'val':
            self.mode = 'test'
            for file_name in file_names:
                self.train.append(os.path.join(mypath, self.mode, 'composite_images', file_name))
                self.mask.append(os.path.join(mypath, self.mode, 'masks', file_name))
                self.anno.append(os.path.join(mypath, self.mode,'real_images', file_name))

        self.trans = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

        print('total Dataset of ' + self.dataset + ' is : ', len(self.train))
        print(self.mode, len(self.train),
              'mask:', len(self.mask),
              'anno:', len(self.anno)
              )

    def __getitem__(self, item):
        img = Image.open(self.train[item]).convert('RGB')
        mask = Image.open(self.mask[item]).convert('L')
        anno = Image.open(self.anno[item]).convert('RGB')
        

        return {"composite_images": self.trans(img),
                "real_images": self.trans(anno),
                "mask": self.trans(mask),
                "name": self.train[item].split('/')[-1],
                "img_url": self.train[item],
                "mask_url": self.mask[item],
                "anno_url": self.anno[item]
                }

    def __len__(self):
        return len(self.train)



























