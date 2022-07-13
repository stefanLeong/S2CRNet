import torch
import os
import numpy as np
import cv2
# fo = open('/home/ipprlab/Projects/s2am_extention/requirements.txt', mode='r+')
# for line in fo.readlines():
#     list = line.split('==')
#     print(list[0])
#     fo.write(list[0] + '\n')

def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    for mask_item in os.listdir(masks_path):
        print(mask_item)
        mask_path = os.path.join(masks_path, mask_item)  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        img_path = os.path.join(images_path, mask_item[:-6] + '.jpg')
        img = cv2.imread(img_path)
        print(mask_path, img_path)
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, mask_item), masked)

images_path = '/home/ipprlab/Projects/s2am_extention/datasets/HCOCO/composite_images/'
masks_path = '/home/ipprlab/Projects/s2am_extention/datasets/HCOCO/masks/'
masked_path = '/home/ipprlab/Projects/s2am_extention/datasets/HCOCO/fake_region'

def add_mask2image_binary1(images_path, masks_path, masked_path):
# Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-6]+'.png')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, img_item), masked)

add_mask2image_binary1(images_path, masks_path, masked_path)
# img_item = 'c196_1541299.png'
# print(img_item.split('_')[0], img_item)
