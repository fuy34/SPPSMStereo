import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np
import glob
import random

def dataloader(filepath, typ = 'train', b_HRonly=False):
  total_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  train_list = ['Adirondack', 'Jadeplant', 'Motorcycle', 'Piano', 'Pipes',
                 'Playroom', 'Playtable', 'Recycle', 'Shelves', 'Vintage']

  # train_list = ['Adirondack', 'Jadeplant', 'Motorcycle', 'Piano', 'Pipes',
  #               'Playroom', 'Playtable', 'Recycle', 'Shelves', 'Vintage',
  #               'Cable', 'Bicycle1','Backpack', 'Classroom1','Couch',
  #               'Flowers','Mask','Sword1','Shopvac', 'Sticks', 'Umbrella','Storage']

  if typ == 'train':
    img_list = [elem for elem in total_list if elem.split('-')[0] in train_list]
    val = [elem for elem in total_list if elem.split('-')[0] not in train_list]

    left_val = ['%s/%s/im0.png' % (filepath, img) for img in val]
    right_val = ['%s/%s/im1.png' % (filepath, img) for img in val]
    disp_val_L = ['%s/%s/disp0GT.pfm' % (filepath, img) for img in val]
    disp_val_R = ['%s/%s/disp1GT.pfm' % (filepath, img) for img in val]

  elif typ == 'trainval':
      if b_HRonly:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + '/HRVS_train.txt') as tf:
          train_list = tf.readlines()
          img_list = [elem[:-1] for elem in train_list]

        with open(dir_path + '/HRVS_val.txt') as vf:
          val_list = vf.readlines()
          val = [elem[:-1] for elem in val_list]
        left_val = ['%s/%s/im0.png' % (filepath, img) for img in val]
        right_val = ['%s/%s/im1.png' % (filepath, img) for img in val]
        disp_val_L = ['%s/%s/disp0GT.pfm' % (filepath, img) for img in val]
        disp_val_R = ['%s/%s/disp1GT.pfm' % (filepath, img) for img in val]

      else:
        img_list = total_list
        left_val, right_val, disp_val_L = [], [], []
  else:
    print("Wrong data type")
    exit(1)
    # print(img_list, total_list)

  left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]
  disp_train_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in img_list]


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

# def dataloader(filepath, typ = 'train'):
#   total_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]
#
#   train_list = ['Adirondack', 'Jadeplant', 'Motorcycle', 'Piano', 'Pipes',
#                  'Playroom', 'Playtable', 'Recycle', 'Shelves', 'Vintage']
#
#   if typ == 'train':
#     img_list = [elem for elem in total_list if elem.split('-')[0] in train_list]
#   elif typ == 'trainval':
#     img_list = total_list
#
#   val = [elem for elem in total_list if elem.split('-')[0] not in train_list]
#
#   # print(img_list, val)
#
#   left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]
#   right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
#   disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]
#   disp_train_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in img_list]
#
#   left_val = ['%s/%s/im0.png'% (filepath,img) for img in val]
#   right_val =['%s/%s/im1.png'% (filepath,img) for img in val]
#   disp_val_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in val]
#   disp_val_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in val]
#
#   return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L