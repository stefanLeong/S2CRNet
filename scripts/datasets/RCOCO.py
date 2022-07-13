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
import cv2
from scripts.utils.osutils import *
from scripts.utils.imgUtils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as tf


class RCOCO(data.Dataset):
    def __init__(self, mode, args=None, sample=[], gan_norm=False):
        self.train = []
        self.anno = []
        self.mask = []
        self.args = args
        self.input_size = args.input_size
        self.base_folder = args.base_dir
        self.dataset = args.data
        self.file_names = []
        self.labels = []
        self.isTrain = False if self.dataset.find('train') == -1 else True
        mypath = join(self.base_folder, self.dataset)
        if mode == 'train':
            print('loading training file...')
            self.train_file = join(mypath, self.dataset) + '_train_slabels_04.txt'

        elif mode == 'val':
            print('loading validation file...')
            self.train_file = join(mypath, self.dataset) + '_test_slabels_04.txt'

        with open(self.train_file, 'r') as f:
            print(self.args.withLabel)
            if self.args.withLabel == 'True':
                for line in f.readlines():
                    self.file_names.append(line.rstrip().split(' ')[0])
                    self.labels.append(int(line.rstrip().split(' ')[2]))
            elif self.args.withLabel == 'False':
                for line in f.readlines():
                    self.file_names.append(line.rstrip().split(' ')[0])
                    self.labels.append(int(1.0))
            else:
                print("Args--withLabel input error!")

        for file_name in self.file_names:
            self.train.append(os.path.join(mypath, 'composite_images', file_name))
            name_part = file_name.split('.')[0]
            mask_path = os.path.join(mypath, 'masks', name_part + '_mask.jpg')
            self.mask.append(mask_path)
            anno_path = os.path.join(mypath, 'real_images', name_part + '_gt.jpg')
            self.anno.append(anno_path)



        self.trans = transforms.Compose([
            transforms.Resize((self.args.hr_size, self.args.hr_size)),
            transforms.ToTensor()
        ])

        self.resize_trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        print('total Dataset of ' + self.dataset + ' is : ', len(self.train))
        print('train:', len(self.train),
              'mask:', len(self.mask),
              'anno:', len(self.anno)
              )

    def __getitem__(self, item):

        img = Image.open(self.train[item]).convert('RGB')


        cv2_img = cv2.imread(self.train[item])
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img = cv2.resize(cv2_img, (256, 256))

        # cv2_img = Image.fromarray(cv2_img)
        # img = Image.fromarray(img)
        mask = Image.open(self.mask[item]).convert('L')
        # mask = cv2.imread(self.mask[item])
        # mask = mask[:, :, 0].astype(np.float32) / 255.
        # mask = mask.astype(np.uint8)
        # mask = Image.fromarray(mask)

        # anno = cv2.imread(self.anno[item])
        # anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
        # anno = Image.fromarray(anno)
        bbox = mask.getbbox()  # (left, upper, right, lower)
        anno = Image.open(self.anno[item]).convert('RGB')
        thumb_foreground = img.crop(bbox)
        thumb_mask = mask.crop(bbox)

        if self.isTrain and random.random() > 0.5:
            img, anno, mask, thumb_foreground, thumb_mask = [item.transpose(Image.FLIP_LEFT_RIGHT) for item in
                                                             [img, mask, thumb_foreground, thumb_mask]]
        return {"composite_images": self.resize_trans(img),
                "mask": self.resize_trans(mask),
                "real_images": self.resize_trans(anno),
                "fore_images": self.resize_trans(thumb_foreground),
                "fore_mask": self.resize_trans(thumb_mask),
                "label": self.labels[item],
                "name": self.train[item].split('/')[-1],
                "img_url": self.train[item],
                "mask_url": self.mask[item],
                "ori_img": self.trans(img),
                "cv2_img": tf.to_tensor(cv2_img)
                # "ori_mask": self.trans(mask),
                # "ori_tar": self.trans(anno)
                }

    def __len__(self):
        return len(self.train)

    def get_params(self, size):
        w, h = size
        new_h = h
        new_w = w
        if self.args.preprocess == 'resize_and_crop':
            new_h = new_w = self.args.load_size
        elif self.args.preprocess == 'scale_width_and_crop':
            new_w = self.args.load_size
            new_h = self.args.load_size * h // w

        x = random.randint(0, np.maximum(0, new_w - self.args.crop_size))
        y = random.randint(0, np.maximum(0, new_h - self.args.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def get_transform(self, params=None, method=Image.BICUBIC, convert=True):
        transform_list = []
        if 'resize' in self.args.preprocess:
            osize = [self.args.input_size, self.args.input_size]
            transform_list.append(transforms.Resize(osize, method))

        if not self.args.no_flip:
            if params['flip']:
                transform_list.append(transforms.Lambda(lambda img: self.__flip(img)))

        if convert:
            transform_list += [transforms.ToTensor()]

        return transforms.Compose(transform_list)






























