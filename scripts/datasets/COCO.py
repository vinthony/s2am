from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import h5py
from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imutils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter

class COCO(data.Dataset):
    def __init__(self,train,config=None, sample=[],gan_norm=False):

        self.train = []
        self.anno = []
        self.mask = []
        self.input_size = config.input_size
        self.normalized_input = config.normalized_input
        self.base_folder = config.base_dir
        self.dataset = train+config.data

        if config == None:
            self.data_augumentation = False
            self.seg = False
        else:
            self.data_augumentation = config.data_augumentation
            self.seg = config.withseg

        self.istrain = False if self.dataset.find('train') == -1 else True
        self.sample = sample
        self.gan_norm = gan_norm
        mypath = join(self.base_folder,self.dataset)
        file_names = [f for f in sorted(listdir(join(mypath,'image'))) if isfile(join(mypath,'image', f)) ]

        if config.limited_dataset > 0:
            file_names = file_names[0:config.limited_dataset]
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(mypath,'image',file_name))
            self.mask.append(os.path.join(mypath,'mask',file_name))
            self.anno.append(os.path.join(mypath,'natural',file_name))

        if len(self.sample) > 0 :
            self.train = [ self.train[i] for i in self.sample ] 
            self.mask = [ self.mask[i] for i in self.sample ] 
            self.anno = [ self.anno[i] for i in self.sample ] 

        print('total Dataset of '+self.dataset+' is : ', len(self.train))


    def __getitem__(self, index):
      
        img_path = self.train[index]
        mask_path = self.mask[index]
        anno_path = self.anno[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        anno = Image.open(anno_path).convert('RGB')

        if self.seg:
            seg = Image.open(mask_path.replace('mask','anno')).convert('L')

        # transform segmentation mask to  

        trans = transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor()
            ])

        # online data_augmentation here
        # for the splicing regions, we transform this part with simple image processing technique
        if self.istrain and self.data_augumentation:
            if random.random() < 0.5 : anno = ImageEnhance.Color(anno).enhance(random.uniform(0.5,1))
            if random.random() < 0.5 : anno = ImageEnhance.Contrast(anno).enhance(random.uniform(0.5,1))
            if random.random() < 0.5 : anno = ImageEnhance.Brightness(anno).enhance(random.uniform(0.5,1))

            img.paste(anno,mask=mask)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                anno = anno.transpose(Image.FLIP_LEFT_RIGHT)

        # composite image transfrom

        mask = trans(mask)

        if self.normalized_input:
            trans1 = transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
        
            img_origin = trans(img)
            img_norm = trans1(img)
            img = torch.cat((img_origin,img_norm),0)
        elif self.gan_norm:
            trans1 = transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])

            img = trans1(img)
        else:
            img = trans(img)

        anno = trans(anno)

        if self.seg:
            seg = trans(seg).long()
            anno = (anno,seg)
        else:
            anno = (anno,anno)

        inputs = torch.cat([img,mask],0)

        return (inputs,anno)

    def __len__(self):

        return len(self.train)
