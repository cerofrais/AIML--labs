import collections
import os.path as osp

import numpy as np
import PIL.Image
import cv2
import scipy.io
import torch
from torch.utils import data
from random import shuffle
import os.path

import os
import sys
import copy
import time
import math
import matplotlib.pyplot as plt

class custom_data_loader(data.Dataset):    ### custom data loader
    
    def __init__(self, img_root, image_list, mirror = True, crop = True, crop_shape = [256, 256],
                 resize=False, resize_shape=[128,128], split = 'train', classes = 'IMFDB'):

    	self.img_root = img_root
        self.split = split
        
        self.image_list = [line.rstrip('\n') for line in open(image_list)]
        
        self.classes = None
        if classes == 'IIC':
            self.classes = ['baba_ramdev', 'biswa',  'dhinchak_pooja',  'khali',  'priya_prakash']
        elif classes == 'IMFDB':
            self.classes = ['AamairKhan', 'Rimisen', 'Kajol', 'KareenaKapoor','RishiKapoor', 'AmrishPuri', 'AnilKapoor', 'AnupamKher', 'BomanIrani', 'HrithikRoshan', 'KajalAgarwal', 'KatrinaKaif', 'Madhavan', 'MadhuriDixit', 'Umashri', 'Trisha']

        self.mirror = mirror
        self.crop = crop
        self.crop_shape = crop_shape
        
        self.resize = resize
        self.resize_shape = resize_shape

        #self.mean_bgr = np.array([123.68, 116.779, 103.939])   ### mean BGR of ImageNet
        self.mean_bgr = np.array([0])   ### lightnet does not perform mean subtraction of input

        self.files = collections.defaultdict(list)
        for f in self.image_list:
            self.files[self.split].append({'img': img_root+f, 'lbl': 0})
        
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        
        image_file_name = self.img_root + self.image_list[index]
        
        image = None
        if os.path.isfile(image_file_name):
            image = cv2.imread(image_file_name, 0)  ### 0 indicates load image as grayscale image
        if image is None:
            print 'creating a default image'
            image = np.zeros((self.resize_shape[1], self.resize_shape[0]))
            if self.resize :
                image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
            
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
            if self.mirror:
                flip = np.random.choice(2)*2-1    ### randomly flip the image
                image = image[:, ::flip, :]

            if self.crop:
                image = self.get_random_crop(image, self.crop_shape)

            label = [self.classes.index(i) for i in self.classes if self.image_list[index].split('/')[0] == i][0]
        
            l = torch.IntTensor(1, 1)
            l = label

            return self.transform_image(image), l
            
        else :    
            if self.resize :
                print 'image found resizing'
                image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
            
                image = image.reshape(image.shape[0], image.shape[1], 1)
            
            if self.mirror:
                flip = np.random.choice(2)*2-1    ### randomly flip the image
                image = image[:, ::flip, :]

            if self.crop:
                image = self.get_random_crop(image, self.crop_shape)

            label = [self.classes.index(i) for i in self.classes if self.image_list[index].split('/')[0] == i][0]
        
            l = torch.IntTensor(1, 1)
            l = label

            return self.transform_image(image), l

    def transform_image(self, image):
        image = image.astype(np.float64)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()   ### convert from numpy array to torch tensor

        return image


    def get_random_crop(self, im, crop_shape):
        """
        crop shape is of the format: [rows cols]
        """
        r_offset = 0
        c_offset = 0
        
        if im.shape[0] == crop_shape[0]:
            r_offset = 0
        else:
            r_offset = np.random.randint(0, im.shape[0] - crop_shape[0] + 1)
            
        if im.shape[1] == crop_shape[1]:
            c_offset = 0
        else:
            c_offset = np.random.randint(0, im.shape[1] - crop_shape[1] + 1)

        crop_im = im[r_offset: r_offset + crop_shape[0], c_offset: c_offset + crop_shape[1], :]

        return crop_im