from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess
from models import *
import shutil
import cv2
from PIL import Image
from utils.save_res import save_pfm
from utils.preprocess import get_transform
from skimage.transform import rescale, resize
#cudnn.benchmark = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/home/fuy34/stereo_data/middlebury/mb-ex/trainingF/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=768,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--savepath', metavar='DIR', default= '/home/fuy34/Dropbox/Fix_ori_sp4pre' ,
                    help='save path')

parser.add_argument('--sz_list', type=float, default= [4], #, 8, 16
                    help='spixel loss weight')
parser.add_argument('--train_img_height', '-t_imgH', default=256*4, #384,
                    type=int,  help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default= 512*4, #768,
                    type=int, help='img width')

parser.add_argument('--val_img_height', '-v_imgH', default=2048, #512 368
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--val_img_width', '-v_imgW', default=3072,   #960 1232
                    type=int, help='img width must be 16*n')

parser.add_argument('--batchsize', type=int, default=4, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--test_batchsize', type=int, default=4, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# dataloader
from dataloader import listfiles as DA
_, _, _, test_left_img, test_right_img, _ = DA.dataloader(args.datapath, typ='train')

# construct model
model = prePSMNet_small(args, None, b_finetune= False)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrained_dict['state_dict'])
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# def test(imgL, imgR):
#     # model.eval()
#     #
#     # if args.cuda:
#     #     imgL = torch.FloatTensor(imgL).cuda()
#     #     imgR = torch.FloatTensor(imgR).cuda()
#
#     imgL, imgR = Variable(imgL), Variable(imgR)
#
#     with torch.no_grad():
#         output = model(imgL, imgR)
#     output = torch.squeeze(output)
#     pred_disp = output.data.cpu().numpy()
#
#     return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    model.eval()
    for inx in range(len(test_left_img)):
        print(test_left_img[inx])
        # if "Shopvac" in test_left_img[inx]:
        #     continue
        if not os.path.isdir(args.savepath):
            os.makedirs(args.savepath)

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        w, h = imgL_o.size
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        print(w, h, imgL.shape)

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, _, _, _, _ = model(imgL, imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h- h
        left_pad  = max_w- w
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
        print(h , w , imgL.shape, pred_disp.shape)
        # save predictions
        idxname = test_left_img[inx].split('/')[-2]
        if not os.path.exists('%s/%s'%(args.savepath,idxname)):
            os.makedirs('%s/%s'%(args.savepath,idxname))

        idxname = '%s/disp0FixOriSP4Pre'%(idxname)
        print('saved to %s/%s' % (args.savepath, idxname))

        # resize to highres
        # pred_disp = cv2.resize(pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        # invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        # pred_disp[invalid] = np.inf

        # np.save('%s/%s-disp.npy'% (args.outdir, idxname.split('/')[0]),(pred_disp))
        # np.save('%s/%s-ent.npy'% (args.outdir, idxname.split('/')[0]),(entropy))
        # cv2.imwrite('%s/%s-disp.png'% (args.outdir, idxname.split('/')[0]),pred_disp/pred_disp[~invalid].max()*255)
        # cv2.imwrite('%s/%s-ent.png'% (args.outdir, idxname.split('/')[0]),entropy/entropy.max()*255)
        # cv2.imwrite('%s/%s-disp.png' % (args.savepath, idxname.split('/')[0]), (100 * pred_disp).astype(np.uint16))
        with open('%s/%s.pfm'% (args.savepath, idxname),'w') as f:
            save_pfm(f,pred_disp[::-1,:])
        with open('%s/%s/timeFixOriSP4Pre.txt'%(args.savepath,idxname.split('/')[0]),'w') as f:
             f.write(str(ttime))
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

