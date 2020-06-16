from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
from collections import namedtuple
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

        resize = transforms.Resize((self.input_size,self.input_size))
        trans.append(resize)
        transimage.append(resize)

        trans.append(transforms.ToTensor())
        transimage.append(transforms.ToTensor())
        
        if isfile(self.train[index].replace('image','natural')):
            target = transforms.Compose(transimage)(target)
        else:
            target = transforms.Compose(trans)(target)
            
        img = transforms.Compose(transimage)(img)
        mask = transforms.Compose(trans)(mask)
        inputs = torch.cat([img,mask],0)

        return (inputs,target,img_path)

    def __len__(self):

        return len(self.train)
