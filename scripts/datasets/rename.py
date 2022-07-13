import cv2
import os
from scripts.utils.osutils import *
from scripts.utils.transforms import *
from os.path import isfile, join
from scripts.utils.imgUtils import *
img_path = '/home/ipprlab/projects/datasets/Hday2night/real_images'
mask_path = '/home/ipprlab/projects/datasets/Hday2night/real_images'
new_imgpath = '/home/ipprlab/projects/datasets/HAdobe5k/test/real_images/'
image_txt = '/home/ipprlab/projects/datasets/HAdobe5k/HAdobe5k_test.txt'

real_images_list = os.listdir(mask_path)

# f = open(image_txt, 'r')
# train = open('/home/ipprlab/projects/datasets/Hday2night/' + 'train.txt', 'a+')
# list1 = f.readlines()
# train_list = []
# print(len(list1))
# for item in list1:
#     name = item.split('_')[0]
#     train_list.append(name+ '.jpg')
#
# print(len(train_list))
# train_list = list(set(train_list))
# for item in train_list:
#     train.write(item + '\n')

# for line in f.readlines():
#     print(join('/home/ipprlab/projects/datasets/HFlickr/', 'real_images', line))
#     if isfile(join('/home/ipprlab/projects/datasets/HFlickr/', 'real_images', line)):
#         print(True)
# print(sorted([line for line in f.readlines() if isfile(join('/home/ipprlab/projects/datasets/HFlickr/', 'real_images', line))]))

# name = list[0].split('.')[0]
# print(name[:-4])
# train.write(name[:-4] + '\n')
# train.write(name[:-4])
# for item in list:
# f = open(image_txt, 'r+')
# list1 = f.readlines()
# for item in os.listdir(img_path):
#     new_name = item.split('.')[0]
#     for mask in masks_list:
#         img = cv2.imread(os.path.join(mask_path, mask))
#         mname = mask.split('.')[0]
#         if new_name.find(mname) > -1:
#             print(new_name + '.jpg')
#             cv2.imwrite(os.path.join(new_imgpath, new_name + '.jpg'), img)

img_path = "/home/ipprlab/projects/DeepHarmonization/result/"
f = open(image_txt, 'r+')
list1 = os.listdir(img_path)
print(len(list1))
for item in list1:
    img = cv2.imread(os.path.join(img_path, item))
    img = cv2.resize(img, (256, 256))
    cv2.imwrite("/home/ipprlab/projects/DeepHarmonization/256/" + item, img)

