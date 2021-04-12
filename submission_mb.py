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

from utils.save_res import save_pfm
from utils.preprocess import get_transform
from skimage.transform import rescale, resize
#cudnn.benchmark = True
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default= '/home/fuy34/stereo_data/middlebury/mb-ex/trainingF/', #'/home/fuy34/Dropbox/middlebury/final_submission/MiddEval3/trainingF', #
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
parser.add_argument('--savepath', metavar='DIR', default= '/home/fuy34/Dropbox/joint_sp16_new',#'/home/fuy34/Dropbox/joint_sp16_final_trainE25' ,
                    help='save path')

parser.add_argument('--sz_list', type=float, default= [16], #, 8, 16
                    help='spixel loss weight')
parser.add_argument('--train_img_height', '-t_imgH', default=256*4, #384,
                    type=int,  help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default= 512*4, #768,
                    type=int, help='img width')

parser.add_argument('--val_img_height', '-v_imgH', default=2048, #512 368
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--val_img_width', '-v_imgW', default=3072,   #960 1232
                    type=int, help='img width must be 16*n')

parser.add_argument('--batchsize', type=int, default=1, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--test_batchsize', type=int, default=1, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# dataloader
from dataloader import listfiles as DA
_, _, _, test_left_img, test_right_img, _ = DA.dataloader(args.datapath, typ='train')
# test_left_img, test_right_img, _, _, _, _, = DA.dataloader(args.datapath, typ='trainval')
# print(test_left_img)
# construct model
model = prePSMNet_small(args, None, b_finetune= True)
# model = spPSMNet(args)
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

        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))[:,:,:3]
        # convert to quater resolution
        h, w, _ = imgL_o.shape
        # imgL_o = resize(imgL_o, (h // 4, w // 4), order=0)
        # imgR_o = resize(imgR_o, (h // 4, w // 4), order=0)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        print(w, h, imgL.shape)

        # imgsize = imgL_o.shape[:2]
        #
        # if args.max_disp>0:
        #     max_disp = int(args.max_disp)
        # else:
        #     with open(test_left_img[inx].replace('im0.png','calib.txt')) as f:
        #         lines = f.readlines()
        #         max_disp = int(int(lines[6].split('=')[-1]))
        #         # max_disp = min(384, max_disp) #add me
        #
        # ## change max disp
        # tmpdisp = int(max_disp*args.testres//64*64)
        # if (max_disp*args.testres/64*64) > tmpdisp:
        #     model.module.maxdisp = tmpdisp + 64
        # else:
        #     model.module.maxdisp = tmpdisp
        # if model.module.maxdisp ==64: model.module.maxdisp=128
        # model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
        # model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
        # model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
        # model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
        # print(model.module.maxdisp)
        
        # resize
        # imgL_o = cv2.resize(imgL_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        # imgR_o = cv2.resize(imgR_o,None,fx=args.testres,fy=args.testres,interpolation=cv2.INTER_CUBIC)
        # imgL = processed(imgL_o).numpy()
        # imgR = processed(imgR_o).numpy()
        #
        # imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        # imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

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
            pred_disp, _, _, _ = model(imgL, imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

        # save predictions
        idxname = test_left_img[inx].split('/')[-2]
        if not os.path.exists('%s/%s'%(args.savepath,idxname)):
            os.makedirs('%s/%s'%(args.savepath,idxname))

        idxname = '%s/disp0SPPSMNet_15'%(idxname)
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

        with open('%s/%s.pfm'% (args.savepath, idxname),'w') as f:
            save_pfm(f,pred_disp[::-1,:])
        with open('%s/%s/timeSPPSMNet_15.txt'%(args.savepath,idxname.split('/')[0]),'w') as f:
             f.write(str(ttime))
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

