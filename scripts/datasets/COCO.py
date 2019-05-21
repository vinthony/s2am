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
    def __init__(self,train,args=None, sample=[]):

        self.train = []
        self.input_size = args.input_size
        self.base_folder = args.base_dir
        self.dataset = train+args.data
        self.args = args

        self.istrain = False if self.dataset.find('train') == -1 else True
        self.sample = sample
        mypath = join(self.base_folder,self.dataset)
        file_names = [f for f in sorted(listdir(join(mypath,'image'))) if isfile(join(mypath,'image', f)) ]

        if args.limited_dataset > 0:
            file_names = file_names[0:args.limited_dataset]
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(mypath,'image',file_name))

        if len(self.sample) > 0 :
            self.train = [ self.train[i] for i in self.sample ] 
           

        print('total Dataset of '+self.dataset+' is : ', len(self.train))


    def __getitem__(self, index):
      
        img_path = self.train[index]
        img = Image.open(img_path).convert('RGB')

        mask_path = self.train[index].replace('image','mask')
        mask = Image.open(mask_path).convert('L')

        if isfile(self.train[index].replace('image','natural')):
            target = Image.open(self.train[index].replace('image','natural')).convert('RGB')
        else:
            target = img.copy()


        trans = []
        transimage = []

        if self.args.resize_and_crop == 'resize':
            resize = transforms.Resize((self.input_size,self.input_size))
            trans.append(resize)
            transimage.append(resize)

        if self.args.resize_and_crop == 'crop':
            # if image < self.input_size,padding with zero
            imh,imw = img.size
            
            rdh = random.randint(0,(imh - self.input_size)//2) if imh - self.input_size > 2 else 0
            rdw = random.randint(0,(imw - self.input_size)//2) if imw - self.input_size > 2 else 0

            coor = (rdw,rdh,rdw+self.input_size,rdh+self.input_size)

            img = img.crop(coor)
            mask = mask.crop(coor)
            target = target.crop(coor)

        trans.append(transforms.ToTensor())
        transimage.append(transforms.ToTensor())

        if self.args.norm_type == 'gan':
            norm = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            transimage.append(norm)
            
        if self.args.norm_type == 'vgg':
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            transimage.append(norm)
        
        if isfile(self.train[index].replace('image','natural')):
            target = transforms.Compose(transimage)(target)
        else:
            target = transforms.Compose(trans)(target)
            
        img = transforms.Compose(transimage)(img)
        mask = transforms.Compose(trans)(mask)
        inputs = torch.cat([img,mask],0)

        return (inputs,target,img_path.split('/')[-1])

    def __len__(self):

        return len(self.train)
