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

class HCOCO(data.Dataset):
    def __init__(self, mode, args=None, sample=[], gan_norm=False):
        self.train = []
        self.anno = []
        self.mask = []

        self.input_size = args.input_size
        self.base_folder = args.base_dir
        self.dataset = args.data
        self.file_names = []
       
        self.isTrain = False if self.dataset.find('train') == -1 else True
        mypath = join(self.base_folder, self.dataset)
        if mode == 'train':
            print('loading training file...')
            self.train_file = join(mypath, self.dataset) + '_train.txt'
           
        elif mode == 'val':
            print('loading validation file...')
            self.train_file = join(mypath, self.dataset) + '_test.txt'
            
            
        with open(self.train_file, 'r') as f:
            for line in f.readlines():
                self.file_names.append(line.rstrip())

        for file_name in self.file_names:
            self.train.append(os.path.join(mypath, 'composite_images', file_name))
            name_parts = file_name.split('_')
            mask_path = os.path.join(mypath, 'masks', file_name)
            mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
            self.mask.append(mask_path)
            anno_path = os.path.join(mypath, 'real_images', file_name)
            anno_path = anno_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
            self.anno.append(anno_path)

        self.trans = transforms.Compose([
#             transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

        print('total Dataset of ' + self.dataset + ' is : ', len(self.train))
        print('train:', len(self.train),
              'mask:', len(self.mask),
              'anno:', len(self.anno)
              )

    def __getitem__(self, item):

        img = Image.open(self.train[item]).convert('RGB')
        mask = Image.open(self.mask[item]).convert('L')
        anno = Image.open(self.anno[item]).convert('RGB')

        # trans_params = self.get_params(img.size)

        # if self.mode == 'val':
        #     self.trans = transforms.Compose([
        #         transforms.ToTensor()
        #     ])
        # elif self.mode == 'train':
        #     self.trans = self.get_transform(trans_params)

        if self.isTrain and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            anno = anno.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"composite_images": self.trans(img),
                "full_images": transforms.ToTensor()(img),
                "real_images": self.trans(anno),
                "mask": self.trans(mask),
                "name": self.train[item].split('/')[-1],
                "img_url": self.train[item],
                "mask_url": self.mask[item],
                "anno_url": self.anno[item]
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

class SEMI(data.Dataset):
    def __init__(self, mode, args=None, sample=[], gan_norm=False):
        self.train = []
        self.anno = []
        self.mask = []

        self.input_size = args.input_size
        self.base_folder = args.base_dir
        self.dataset = args.data
        self.file_names = []
       
        self.isTrain = False if self.dataset.find('train') == -1 else True
        mypath = join(self.base_folder, self.dataset)
        if mode == 'train':
            self.dataset = 'HCOCO'
            mypath = '/data/Datasets/HCOCO'
            print('loading training file...')
            self.train_file = join(mypath, self.dataset) + '_train.txt'
           
        elif mode == 'val':
            print('loading validation file...')
            self.train_file = join(mypath, self.dataset) + '_test.txt'
            
            
        with open(self.train_file, 'r') as f:
            for line in f.readlines():
                self.file_names.append(line.rstrip())

        for file_name in self.file_names:
            self.train.append(os.path.join(mypath, 'images', file_name))
            name_parts = file_name.split('_')
            mask_path = os.path.join(mypath, 'masks', file_name)
            mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
            self.mask.append(mask_path)
            anno_path = os.path.join(mypath, 'reals', file_name)
            anno_path = anno_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
            self.anno.append(anno_path)

        self.trans = transforms.Compose([
            # transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

        print('total Dataset of ' + self.dataset + ' is : ', len(self.train))
        print('train:', len(self.train),
              'mask:', len(self.mask),
              'anno:', len(self.anno)
              )

    def __getitem__(self, item):
        
        # random choice another images
        ix = np.random.choice(len(self.train),1)[0]
        
        mask = Image.open(self.mask[ix]).convert('L').resize((256,256))
        anno_r = Image.open(self.anno[ix]).convert('RGB').resize((256,256))

        # mask = Image.open(self.mask[item]).convert('L').resize((256,256))
        anno = Image.open(self.anno[item]).convert('RGB').resize((256,256))

        img = Image.composite(anno_r, anno, mask)

        # anno.paste(anno_r,(0, 0),mask_r)

        # img.save('img.png')
        # anno_r.save('annor')

        # import pdb; pdb.set_trace()


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



























