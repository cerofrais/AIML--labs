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

class siamese_data_loader(data.Dataset):
    
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
        
        index = np.random.randint(0, len(self.image_list))
        
        image_file_name_1 = self.img_root + self.image_list[index]
        i1 = min(index-300, (index+300)%len(self.image_list))
        i2 = max(index-300, (index+300)%len(self.image_list))
        
        if i1==i2:
            i2+=1
        index2 = np.random.randint(i1, i2)   ### get a random pair. There are much better ways to do this.
        if index2==index:
            index2 -= np.random.randint(0,100)
        #print index, index2
        image_file_name_2 = self.img_root + self.image_list[index2]
        
        image_1 = None
        if os.path.isfile(image_file_name_1):
            image_1 = cv2.imread(image_file_name_1, 0)  ### 0 indicates load image as grayscale image

        if image_1 is None:
            #print 'ERROR: couldn\'t find image -> ', image_file_name_1
            print 'creating a default image'
            image_1 = np.zeros((self.resize_shape[1], self.resize_shape[0]))
            if self.resize:
                image_1 = cv2.resize(image_1, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
        
            image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1], 1)
            
            if self.mirror:
                flip = np.random.choice(2)*2-1
                image_1 = image_1[:, ::flip, :]

            if self.crop:
                image_1 = self.get_random_crop(image_1, self.crop_shape)

            label_1 = [self.classes.index(i) for i in self.classes if self.image_list[index].split('/')[0] == i][0]
        
        else :
        
            if self.resize:
                image_1 = cv2.resize(image_1, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
        
            image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1], 1)
            if self.mirror:
                flip = np.random.choice(2)*2-1
                image_1 = image_1[:, ::flip, :]

            if self.crop:
                image_1 = self.get_random_crop(image_1, self.crop_shape)

            label_1 = [self.classes.index(i) for i in self.classes if self.image_list[index].split('/')[0] == i][0]
        
        
        image_2 = None
        if os.path.isfile(image_file_name_2):
            image_2 = cv2.imread(image_file_name_2, 0)  ### 0 indicates load image as grayscale image

        if image_2 is None:
        
            #print 'ERROR: couldn\'t find image -> ', image_file_name_2
            image_2 = np.zeros((self.resize_shape[1], self.resize_shape[0]))
            if self.resize:
                image_2 = cv2.resize(image_2, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
        
            image_2 = image_2.reshape(image_2.shape[0], image_2.shape[1], 1)
            if self.mirror:
                flip = np.random.choice(2)*2-1
                image_2 = image_2[:, ::flip, :]

            if self.crop:
                image_2 = self.get_random_crop(image_2, self.crop_shape)

            label_2 = [self.classes.index(i) for i in self.classes if self.image_list[index2].split('/')[0] == i][0]
        
        else : 
            
            if self.resize:
                image_2 = cv2.resize(image_2, (self.resize_shape[1], self.resize_shape[0]))   ### resize_shape is in [rows, cols]
        
            image_2 = image_2.reshape(image_2.shape[0], image_2.shape[1], 1)
            if self.mirror:
                flip = np.random.choice(2)*2-1
                image_2 = image_2[:, ::flip, :]

            if self.crop:
                image_2 = self.get_random_crop(image_2, self.crop_shape)

            label_2 = [self.classes.index(i) for i in self.classes if self.image_list[index2].split('/')[0] == i][0]
        
        
        
        l = torch.FloatTensor(1, 1)
        l = label_1
        
        if label_1 == label_2:              ### same class
            l = 0.0 
        else:                               ### different class
            l = 1.0        
        

        return self.transform_image(image_1), self.transform_image(image_2), l

    def transform_image(self, image):
        image = image.astype(np.float64)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()

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