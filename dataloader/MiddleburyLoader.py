import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
from utils import preprocess
from utils import readpfm as rp
# import dataloader.flow_transforms as flow_transforms
import pdb
import torchvision
import warnings
import random
import cv2

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

IMG_EXTENSIONS = [
 '.jpg', '.JPG', '.jpeg', '.JPEG',
 '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any((filename.endswith(extension) for extension in IMG_EXTENSIONS))


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader,hw_list=[256,512,2000,2960]): #[0.225,0.6], the default is for sceneflow

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.train_h, self.train_w, self.val_h,self.val_w = hw_list[0], hw_list[1], hw_list[2], hw_list[3]

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL[dataL == np.inf] = 0
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        # convert to quater resolution
        # w, h = left_img.size
        # left_img =  left_img.resize((w//4, h//4), resample=Image.NEAREST)
        # right_img = right_img.resize((w // 4, h // 4), resample=Image.NEAREST)
        # dataL = cv2.resize(dataL,(w // 4, h // 4),interpolation=cv2.INTER_NEAREST) /4.

        if self.training:
            w, h = left_img.size
            th, tw =  self.train_h, self.train_w
            # print(left, w, tw, h, th)
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)  # just normalized pic
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            # w, h = left_img.size
            # max_h = int(h // 64 * 64 )
            # max_w = int(w// 64 * 64)
            # if max_h < h: max_h += 64
            # if max_w < w: max_w += 64
            #
            # # print(w, self.val_w, h, self.val_h  )
            # left_img = left_img.crop((w - max_w, h - max_h, w, h))  # will crop from 0, -4, with w, h,  add 0 to the addition area
            # right_img = right_img.crop((w - max_w, h - max_h, w, h))  # in test, the output be croped it
            # dataL = np.lib.pad(dataL, ((max_h - h, 0), (max_w -w ,0 )), mode='constant', constant_values=0)
            #
            # processed = preprocess.get_transform(augment=False)
            # left_img = processed(left_img)
            # right_img = processed(right_img)

            w, h = left_img.size
            max_w, max_h = self.val_w, self.val_h
            # max_h = int(h // 64 * 64 )
            # max_w = int(w// 64 * 64)
            # if max_h < h: max_h += 64
            # if max_w < w: max_w += 64

            # print(w, self.val_w, h, self.val_h  )
            left_img = left_img.crop((w - max_w, h - max_h, w, h))  # will crop from 0, -4, with w, h,  add 0 to the addition area
            right_img = right_img.crop((w - max_w, h - max_h, w, h))  # in test, the output be croped it

            # for h > max_h
            dataL = dataL[h - max_h : h, w - max_w: w]

            # for h < max_h
            # print(left_img.size, dataL.shape)
            # dataL = np.lib.pad(dataL, ((max_h - h, 0), (max_w -w ,0 )), mode='constant', constant_values=0)

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)