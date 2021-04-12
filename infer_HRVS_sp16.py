from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

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
import glob
from utils import readpfm as rp
from utils import init_spixel_grid

import matplotlib.pyplot as plt

from utils.save_res import *
from utils.preprocess import get_transform
from skimage.transform import rescale, resize
#cudnn.benchmark = True

cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/home/fuy34/stereo_data/HR_VS/carla-highres/trainingF/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='prePSMNet_small',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=768,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--savepath', metavar='DIR', default= '/home/fuy34/Dropbox/disp_res/HRVS_joint_sp16_tst' ,
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
_, _, _, test_left_img, test_right_img, disp_true = DA.dataloader(args.datapath, typ='trainval', b_HRonly=True)

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


def main():
    processed = preprocess.get_transform(augment=False)
    model.eval()
    sum_epe = 0
    for inx in range(len(test_left_img)):
        # print(test_left_img[inx])
        name = test_left_img[inx].split('/')[-2]
        # if name not in ['exp-3_w-1_pos-53-76_00350', 'exp-2_w-4_pos-79-14_00400',
        #                 'exp-1_w-8_pos-66-3_00200', 'exp-3_w-1_pos-31-71_00750',
        #                 'exp-0_w-11_pos-19-66_00450'] : continue
        # if name not in ['exp-1_w-8_pos-21-12_00200']: continue #

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

        # print(w, h, imgL.shape)

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
        imgL_in = Variable(torch.FloatTensor(imgL).cuda())
        imgR_in = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, maskL, _, _ = model(imgL_in, imgR_in)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time) #print('time = %.2f' % (ttime*1000) )
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]


        # read gt
        gt_disp = rp.readPFM(disp_true[inx])[0]
        gt_disp[gt_disp == np.inf] = 0
        # cv2.imwrite(args.savepath + '/{}_gt.png'.format(name), (gt_disp * 100).astype(np.uint16))
        gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32)

        mask = (gt_disp != 0)
        err = np.abs(gt_disp - pred_disp)*mask
        epe = np.sum(err)/np.sum(mask)
        sum_epe += epe

        if False : #inx < 50:
            top_cut = 640
            if not os.path.isdir(os.path.join(args.savepath, 'img')):
                os.makedirs(os.path.join(args.savepath, 'img'))
            img_save_path = os.path.join(args.savepath, 'img', name + '.png')
            # shutil.copy(test_left_img[inx], img_save_path)
            img = cv2.imread(test_left_img[inx])
            print(img.shape)
            cv2.imwrite(img_save_path, img[top_cut:])
            # cv2.imwrite(args.savepath + '/{}_err.png'.format(name), (val2uint8(err, 20)))

            MAX_DISP = 500
            MIN_DISP = 0
            # tgt_max = np.max(tgt_disp)
            # tgt_min = np.min(tgt_disp)
            if not os.path.isdir(os.path.join(args.savepath, 'tgt_disp_viz')):
                os.makedirs(os.path.join(args.savepath, 'tgt_disp_viz'))
            tgt_disp_save_name = os.path.join(args.savepath, 'tgt_disp_viz', name + '_gt.png')
            plt.imshow(gt_disp[top_cut:], vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(tgt_disp, MAX_DISP, MIN_DISP)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(tgt_disp_save_name, bbox_inches='tight', pad_inches=0)

            # # save pred disp viz
            if not os.path.isdir(os.path.join(args.savepath, 'pred_disp_viz')):
                os.makedirs(os.path.join(args.savepath, 'pred_disp_viz'))
            pred_disp_viz_save_name = os.path.join(args.savepath, 'pred_disp_viz',  name + '_pred.png')
            plt.imshow( (pred_disp[top_cut:]*mask[top_cut:]), vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(pred_disp_viz_save_name, bbox_inches='tight', pad_inches=0)

            #err
            if not os.path.isdir(os.path.join(args.savepath, 'disp_err')):
                os.makedirs(os.path.join(args.savepath, 'disp_err'))
            disp_save_path = os.path.join(args.savepath, 'disp_err',  name + '_err.png')
            skimage.io.imsave(disp_save_path, val2uint8(err[top_cut:], 20))

            # spxiel
            img_l = torch.FloatTensor(imgL)
            mean_values = torch.tensor([0.485, 0.456, 0.406], dtype=img_l.dtype).view(3, 1, 1).to(img_l.device)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=img_l.dtype).view(3, 1, 1).to(img_l.device)
            img_l = (torch.FloatTensor(imgL) * std + mean_values).cpu().numpy()

            _, _, h, w = imgL.shape

            args.val_img_height ,  args.val_img_width = h, w
            _, spixel_indx, _,_,_ = init_spixel_grid(args, args.sz_list, b_train=False)
            maskL_viz, _ = update_spixl_map(spixel_indx, [maskL])
            spixel_viz_L, _ = get_spixel_image(args, img_l[0].transpose(1, 2, 0), maskL_viz[0].squeeze())
           # print(spixel_viz_L.shape, img_l.shape, top_pad, left_pad)
            spixel_viz_L = spixel_viz_L[:, top_pad:, :-left_pad ]
           # print(spixel_viz_L.shape)

            if not os.path.isdir(args.savepath + '/spixel'):
                os.makedirs(args.savepath + '/spixel')
            dump_path = args.savepath + '/spixel/' + name + '_spixel.png'
            skimage.io.imsave(dump_path, (spixel_viz_L.transpose(1, 2, 0))[top_cut:])

        print('{}: {}'.format(name, epe))

    print('meanEPE: {}'.format(sum_epe / len(test_left_img)))
    torch.cuda.empty_cache()

def val2uint8(mat,maxVal, minVal=0):
    maxVal_mat = np.ones(mat.shape) * maxVal
    minVal_mat = np.ones(mat.shape) * minVal

    mat_vis = np.where(mat > maxVal_mat, maxVal_mat, mat)
    mat_vis = np.where(mat < minVal_mat, minVal_mat, mat_vis)
    return (mat_vis * 255. / maxVal).astype(np.uint8)

if __name__ == '__main__':
    main()

