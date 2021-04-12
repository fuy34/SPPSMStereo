from __future__ import print_function
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from utils import preprocess, readpfm
from utils.save_res import *
from models import *
import shutil
import sys

import matplotlib.pyplot as plt
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='No',
                    help='KITTI version')
parser.add_argument('--datapath', default='/home/fuy34/stereo_data/sceneflow/',
                    help='select model')
parser.add_argument('--loadmodel', default='',
                    help='loading model')
parser.add_argument('--model', default='prePSMNet',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_res', action='store_true', default=False,
                    help='save res')

parser.add_argument('--num_test', type=int, default=100, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of test')
parser.add_argument('--batchsize', type=int, default=1, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--test_batchsize', type=int, default=1, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')

parser.add_argument('--sz_list', type=float, default= [4], #, 8, 16, 32, 64
                    help='spixel loss weight')


parser.add_argument('--train_img_height', '-t_imgH', default=256, #384,
                    type=int,  help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default= 512, #768,
                    type=int, help='img width')

parser.add_argument('--val_img_height', '-v_imgH', default=544, #512
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--val_img_width', '-v_imgW', default=960,   #960
                    type=int, help='img width must be 16*n')


parser.add_argument('--input_img_height', default=544, type=int, #320
                    help='the height to put into network')
parser.add_argument('--input_img_width', default=960, type=int, #448
                    help='the width to put into network')
parser.add_argument('--savepath', metavar='DIR', default= '/home/fuy34/Dropbox/disp_res/fixed_sp16_flyingThings' ,
                    help='save path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


from dataloader import listflowfile as lt

_, _, _, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)


if args.model == 'prePSMNet':
    model = prePSMNet(args, None)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            outputs = model(imgL,imgR)
        output = torch.squeeze(outputs[0])
        pred_disp = output.data.cpu().numpy()
        # maskL, maskR = outputs[1][0], outputs[2][0]
        maskL = outputs[1][0]
        return pred_disp, maskL, None, outputs[3] #, outputs[4]


def main():
    processed = preprocess.get_transform(augment=False)
    EPE = 0
    # num_test = min(len(test_left_img), args.num_test)
    avg_time = 0
    cnt = 0
    # f = open(os.path.join(args.savepath, 'res_192.txt'), "w+")
    # print(test_left_img[0])

    # print('/home/fuy34/stereo_data/sceneflow//FlyingThings3D/frames_cleanpass//TEST/A/0083/left/0007.png' in test_left_img)
    # print('/home/fuy34/stereo_data/sceneflow//FlyingThings3D/frames_cleanpass//TEST/A/0145/left/0007.png' in test_left_img)
    # exit(1)

    for inx in range(0, len(test_left_img), 40):

       name  =test_left_img[inx]

       imgL_o = Image.open(test_left_img[inx]).convert('RGB')
       imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        # print("{}\t\n{}\t\n{}".format(test_left_img[inx],test_right_img[inx],test_left_disp[inx]))
       tgt_disp , scale = readpfm.readPFM(test_left_disp[inx])

       #skip the img have large disp
       # if np.max(tgt_disp) > 192:
       #     f.write("{}:\t skip for large disparity\t\n ".format(img_name))
       #     print("[{}/{}] skip for large disparity".format(inx, len(test_left_img)))
       #     continue

       cnt += 1
       mask =  np.logical_and(tgt_disp > 0, tgt_disp < 192)
       # print(test_left_img[inx].split('/'))
       img_name = "{}_{}_{}".format(test_left_img[inx].split('/')[-4],
                                    test_left_img[inx].split('/')[-3],
                                    test_left_img[inx].split('/')[-1])

       # print(img_name)
       # if name == '/home/fuy34/stereo_data/sceneflow//FlyingThings3D/frames_cleanpass//TEST/A/0083/left/0007.png':
       #     print(img_name)
       #     print(img_name in ['A_0047_0007.png', 'A_0081_0007.png', 'A_0083_0007.png'
       #                     'A_0145_0007.png', 'C_0022_0007.png'])

       if img_name not in ['A_0047_0007.png', 'A_0081_0007.png', 'A_0083_0007.png',
                           'A_0145_0007.png', 'C_0022_0007.png']: continue


       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]]) #just normalize


       # pad to (544, 960)
       top_pad = args.input_img_height - imgL.shape[2]
       left_pad = args.input_img_width - imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp,  maskL, _ , spixel_indx = test(imgL,imgR)
       cost_time = time.time() - start_time
       avg_time += cost_time
       # print('time = %.2f' %(time.time() - start_time))
       # print (pred_disp.shape)


       top_pad   = args.input_img_height - imgL_o.size[1]
       # left_pad  = args.input_img_width - imgL_o.shape[1] #==0
       disp_save = pred_disp[top_pad:,:]

       disp_err = np.abs(tgt_disp - disp_save)
       epe = (disp_err[mask]).mean()
       EPE += epe

       # f.write("[{}/{}] {}: \t EPE: {:.3f} \t time: {:.3f}\t\n ".format(inx,len(test_left_img), img_name, epe, cost_time))
       print("{}: [{}/{}] {}: \t EPE: {:.3f} \t time: {:.3f} ".format(cnt, inx,len(test_left_img), img_name, epe, cost_time))

       if args.save_res:
           # #save img
           if not os.path.isdir(os.path.join(args.savepath, 'img')):
               os.makedirs(os.path.join(args.savepath, 'img'))
           img_save_path = os.path.join(args.savepath, 'img', img_name)
           shutil.copy(test_left_img[inx], img_save_path)

           print(img_name)

           if epe > 2:
               if not os.path.isdir(os.path.join(args.savepath, 'imgR')):
                   os.makedirs(os.path.join(args.savepath, 'imgR'))
               img_save_path = os.path.join(args.savepath, 'imgR', img_name)
               shutil.copy(test_right_img[inx], img_save_path)

           # #save disp
           if not os.path.isdir(os.path.join(args.savepath, 'disp')):
               os.makedirs(os.path.join(args.savepath, 'disp'))
           disp_save_path = os.path.join(args.savepath, 'disp', img_name)
           skimage.io.imsave(disp_save_path,(disp_save*100).astype('uint16'))


           # #save disp
           if not os.path.isdir(os.path.join(args.savepath, 'gt_disp')):
               os.makedirs(os.path.join(args.savepath, 'gt_disp'))
           disp_save_path = os.path.join(args.savepath, 'gt_disp', img_name)
           skimage.io.imsave(disp_save_path, (tgt_disp * 100).astype('uint16'))

           # #save disp err
           if not os.path.isdir(os.path.join(args.savepath, 'disp_err')):
               os.makedirs(os.path.join(args.savepath, 'disp_err'))
           disp_save_path = os.path.join(args.savepath, 'disp_err', img_name)
           skimage.io.imsave(disp_save_path, val2uint8(disp_err,5))

           # get images
           img_l, img_r = torch.FloatTensor(imgL), torch.FloatTensor(imgR)
           mean_values = torch.tensor([0.485, 0.456, 0.406], dtype=img_l.dtype).view(3, 1, 1).to(img_l.device)
           std = torch.tensor([0.229, 0.224, 0.225], dtype=img_l.dtype).view(3, 1, 1).to(img_l.device)
           img_l = (torch.FloatTensor(imgL) * std + mean_values).cpu().numpy()
           img_r = (torch.FloatTensor(imgR) * std + mean_values).cpu().numpy()

            # save spixel
           # print(spixel_indx[0].shape)
           maskL_viz, _ = update_spixl_map(spixel_indx, [maskL])
           # maskR_viz, _ = update_spixl_map(spixel_indx, [maskR])
           spixel_viz_L, _ = get_spixel_image(args, img_l[0].transpose(1, 2, 0), maskL_viz[0].squeeze())
           # spixel_viz_R, _ = get_spixel_image(args, img_r[0].transpose(1, 2, 0), maskR_viz[0].squeeze())
           if not os.path.isdir(args.savepath + '/spixel'):
               os.makedirs(args.savepath + '/spixel')
           dump_path = args.savepath + '/spixel/' + test_left_img[inx].split('/')[-1]
           # print(spixel_viz_L.shape)
           skimage.io.imsave(dump_path, (spixel_viz_L.transpose(1, 2, 0)))
           # skimage.io.imsave(dump_path.replace('.png', '_r.png'), (spixel_viz_R.transpose(1, 2, 0)))

           MAX_DISP = 160
           MIN_DISP = 0
           # tgt_max = np.max(tgt_disp)
           # tgt_min = np.min(tgt_disp)
           if not os.path.isdir(os.path.join(args.savepath, 'tgt_disp_viz')):
               os.makedirs(os.path.join(args.savepath, 'tgt_disp_viz'))
           tgt_disp_save_name = os.path.join(args.savepath, 'tgt_disp_viz', img_name)
           plt.imshow(tgt_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(tgt_disp, MAX_DISP, MIN_DISP)
           plt.axis('off')
           plt.gca().xaxis.set_major_locator(plt.NullLocator())
           plt.gca().yaxis.set_major_locator(plt.NullLocator())
           plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
           plt.margins(0, 0)
           plt.savefig(tgt_disp_save_name, bbox_inches='tight', pad_inches=0)

           # # save pred disp viz
           if not os.path.isdir(os.path.join(args.savepath, 'pred_disp_viz')):
               os.makedirs(os.path.join(args.savepath, 'pred_disp_viz'))
           pred_disp_viz_save_name = os.path.join(args.savepath, 'pred_disp_viz', img_name)
           plt.imshow(disp_save, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
           plt.axis('off')
           plt.gca().xaxis.set_major_locator(plt.NullLocator())
           plt.gca().yaxis.set_major_locator(plt.NullLocator())
           plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
           plt.margins(0, 0)
           plt.savefig(pred_disp_viz_save_name, bbox_inches='tight', pad_inches=0)

           # break
       # if inx >= num_test-1:
       #         break

    print("avg_EPE: {:.3f}, avg_time: {:.3f}, val_num:{}" .format(EPE/ (cnt), avg_time/cnt, cnt))
    # f.write("avg_EPE: {:.3f}, avg_time: {:.3f}, val_num:{}\t\n" .format(EPE/ (cnt), avg_time/cnt, cnt))
    # f.close()

def val2uint8(mat,maxVal, minVal=0):
    maxVal_mat = np.ones(mat.shape) * maxVal
    minVal_mat = np.ones(mat.shape) * minVal

    mat_vis = np.where(mat > maxVal_mat, maxVal_mat, mat)
    mat_vis = np.where(mat < minVal_mat, minVal_mat, mat_vis)
    return (mat_vis * 255. / maxVal).astype(np.uint8)


if __name__ == '__main__':
   main()






