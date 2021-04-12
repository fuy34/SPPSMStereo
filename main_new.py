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
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA

from models import *
from utils import *
import shutil
from glob import glob

'''
--------------------------------------------------
This is the code for the spixel up/downsampling PSMNet (SPPSMNet)
We train the network from the scratch on sceneflow and later fine-tune on different dataset
The spixel size is fixed to 4 on sceneflow, 
the image is downsampled first and fed to PSMNet and then upsampled the cost volume before the disp. regression

Author: Fengting Yang
Final modification: Nov. 2019 


# note
# difference from the previous joint training
# 1. spixel is training from the scratch instead of pre-trained spixel
# 2. the disparity is upsampled instead of cost volume, be same as sp16 case
# 3. the learning schedule become lr: 1e-3 * 10==> 5e-4 * 3 ==> 1e-4 * 3
'''

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='SPPSMNet',
                    help='select model')
parser.add_argument('--datapath', default= '/home/fuy34/stereo_data/sceneflow/', #'/home/fuy34/stereo_data/kitti_2015/training/' ,#
                    help='datapath')
parser.add_argument('--epochs', type=int, default=13,
                    help='number of epochs to train (default:13)')
parser.add_argument('--batchsize', type=int, default=8, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--test_batchsize', type=int, default=8, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--preTrain_spixel', default= './preTrain_spixel_old/',
                    help='preTrain model')
parser.add_argument('--savemodel', default='/data/Fengting/stereo_training/SPPSMNet/useful_sceneflow/', #usefull_res_sceneflow/
                    help='save model')
parser.add_argument('--logname', default='SPPPSM_tst',
                    help='log name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--test_only', action='store_true', default=False,
                    help='test_only')

parser.add_argument('--m_w', type=float, default=15,
                    help='slic position weight')
parser.add_argument('--sp_w', type=float, default=0.1,
                    help='spixel loss weight')
parser.add_argument('--sz_list', type=float, default= [4], #, 8, 16
                    help='spixel loss weight')

parser.add_argument('--recFre', type=int, default=1100,
                    help='recording checkpoint frequence (iter)')
parser.add_argument('--saveFre', type=int, default=2000,
                    help='recording checkpoint frequence (iter)')

parser.add_argument('--train_img_height', '-t_imgH', default=256, #384,
                    type=int,  help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default= 512, #768,
                    type=int, help='img width')

parser.add_argument('--val_img_height', '-v_imgH', default=544, #512
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--val_img_width', '-v_imgW', default=960,   #960
                    type=int, help='img width must be 16*n')

parser.add_argument('--real_img_height', '-r_imgH', default=540, #512 368
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--real_img_width', '-r_imgW', default=960,   #960 1232
                    type=int, help='img width must be 16*n')


parser.add_argument('--epoch_size', default=1e5,   #960
                    type=int, help='img width must be 16*n')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _init_fn(worker_id):
    np.random.seed()
    random.seed()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
train_epoch_size = min(args.epoch_size* args.batchsize, len(all_left_img)) #only for debug
val_epoch_size = min(args.epoch_size* args.batchsize, len(test_left_disp))


TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img[:train_epoch_size],all_right_img[:train_epoch_size],all_left_disp[:train_epoch_size], True),
         batch_size= args.batchsize, shuffle= True, num_workers= 8, drop_last=True, worker_init_fn=_init_fn)

if args.test_only:
    #Note we use 4 gpu, in the last iter some GPU may do not have data, and an error will report, but seems ok
    TestImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
             batch_size= 4, shuffle= False, num_workers= 8, drop_last=False, worker_init_fn=_init_fn)
else:
    TestImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(test_left_img[:val_epoch_size],test_right_img[:val_epoch_size],test_left_disp[:val_epoch_size], False),
             batch_size= args.test_batchsize, shuffle= False, num_workers= 8, drop_last=True, worker_init_fn=_init_fn)

spixel_ckpts = args.preTrain_spixel+'/4.pth.tar'
model = prePSMNet(args, spixel_ckpts, b_pretrain=False)

if args.cuda:
    torch.backends.cudnn.benchmark = True
    model = nn.DataParallel(model)
    model.cuda()

train_params =  [param for name, param in model.named_parameters() if param.requires_grad and not 'spixel' in name]
spixel_params = [param for name, param in model.named_parameters() if param.requires_grad and 'spixel' in name ] #and 'spixel4' not in name

optimizer = optim.Adam( [
                        {'params': train_params},
                        {'params': spixel_params, 'weight_decay': 4e-4} ], lr=0.001, betas=(0.9, 0.999)) #{'params': [s1, s2]},


if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['start_epoch']
    optimizer.load_state_dict(state_dict['optimz_state'])
    best_err = state_dict['best_err']
    total_iters = (start_epoch-1) * len(TrainImgLoader)
else:
    start_epoch = 1
    best_err = 1e4
    total_iters = 0

if args.test_only:
    start_epoch = 1
    args.epochs = 1

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))) #5743010



def train(imgL,imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()


    optimizer.zero_grad()

    outputs = model(imgL, imgR, disp_L, b_joint=True)

    output1 = outputs[0][0]  # torch.squeeze(outputs[0], 1)
    output2 = outputs[1][0]  # torch.squeeze(outputs[1], 1)
    output3 = outputs[2][0]  # torch.squeeze(outputs[2], 1)

    disp_loss = (0.5 * output1.sum() + 0.7 * output2.sum() + output3.sum())/ args.batchsize

    loss_col = 0
    loss_pos = 0
    sp_errs = outputs[3]
    # loss_map_R = outputs[4]

    for i in range(len(args.sz_list)):  # todo 5
        a = sp_errs[0].sum() / args.batchsize  # color loss
        b = args.m_w / (args.sz_list[i]) * sp_errs[1].sum() / args.batchsize #math.sqrt, in our paper we use sqrt, but actually we should not
        loss_col += a
        loss_pos += b
        print("level {}, loss_col: {} loss_pos: {}".format(i, args.sp_w * a, args.sp_w * b))

    spixle_loss = args.sp_w * (loss_col + loss_pos)

    loss = disp_loss + spixle_loss

    viz = {}

    viz['final_output'] = outputs[2][1].cpu().numpy()  # output3.detach().cpu().numpy()
    viz['sp_assign_L'] = [outputs[5]]
    viz["spixel_idx_list"] = outputs[7]

    # for debug only
    # viz['sp_assign_R'] = outputs[6]
    # viz['output1'] = outputs[0][1].cpu().numpy() #output1.detach().cpu().numpy()
    # viz['output2'] = outputs[1][1].cpu().numpy() #output2.detach().cpu().numpy()
    # viz["spImg_l"] = outputs[8][0].numpy() #.detach().cpu().numpy()
    # viz["spImg_r"] = outputs[8][1].numpy() #.detach().cpu().numpy()
    # viz["disturb_img"] = outputs[-1].detach().cpu().numpy()

    viz['mask'] = outputs[-1].cpu().numpy()

    loss.backward()
    optimizer.step()

    return loss.item(), disp_loss.item(), args.sp_w * loss_col.item(), args.sp_w * loss_pos.item(), viz

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.cuda:
            imgL, imgR , disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            mask =  (disp_true < 192)

        output = torch.squeeze(outputs[0], 1)[:,args.val_img_height-540:,:]

        # output = torch.squeeze(outputs[0].data.cpu(),1)[:,args.val_img_height-540:,:] #crop the pad part
        # print(output.shape)

        viz = {}

        viz['final_output'] = output.detach().cpu().numpy()
        viz['mask'] = mask.type(torch.float).detach().cpu().numpy()
        viz['sp_assign_L'] = outputs[1]
        viz["spixel_idx_list"] = outputs[3]

        # torch.cuda.synchronize()
        # viz['sp_assign_R'] = outputs[2]
        # viz["spImg_l"] = outputs[-1][0].detach().cpu().numpy()
        # viz["spImg_r"] = outputs[-1][1].detach().cpu().numpy()

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

        return loss.item(), viz



def adjust_learning_rate(optimizer, epoch):
    # lr = 0.001
    # print(lr)
    if epoch <= args.epochs - 2:
        lr = 1e-3
    elif epoch == args.epochs - 1:
        lr = 5e-4
    else:
        lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

    # if epoch <= 10:
    #     lr = 1e-3
    # elif epoch > 13:
    #     lr = 1e-4
    # else:
    #     lr = 5e-4
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return  lr

def main():
    global best_err, start_epoch, total_iters
    saveName = args.logname + "_spW{}_mW{}_ep{}_b{}".format( args.sp_w, args.m_w,  args.epochs, args.batchsize)
    log = Logger(args.savemodel, name= saveName, s_train = 'train')
    val_log = Logger(args.savemodel, name= saveName , s_train = 'val')

    num_log_img = 4
    print(' total training data: %d; total test data: %d'% (len(TrainImgLoader),len(TestImgLoader)) )
    # with batch size 8  total training data: 4431; total test data: 546

    start_full_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        print('This is %d-th epoch, total iter: %d' %(epoch, len(TrainImgLoader)))
        total_train_loss = 0
        lr = adjust_learning_rate(optimizer,epoch)
        log.scalar_summary('lr', lr, epoch)

        if not args.test_only:
           ## training ##
           start_time = time.time()
           for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

                torch.cuda.synchronize()
                data_time = time.time() - start_time

                loss, disp_loss, spCol_loss, spPos_loss, viz = train(imgL_crop, imgR_crop, disp_crop_L)
                torch.cuda.synchronize()
                print('Epoch [%.2d: %.4d] training loss = %.3f ,  data time = %.2f,  total time = %.2f' %\
                      (epoch, batch_idx, loss, data_time, time.time() - start_time))
                total_train_loss += loss

                if total_iters % 10 == 0:
                    log.scalar_summary('total_loss_batch', loss, total_iters)
                    log.scalar_summary('disp_loss_batch', disp_loss, total_iters)
                    log.scalar_summary('spixel_loss_batch', spCol_loss+spPos_loss, total_iters)
                    log.scalar_summary('col_loss_batch', spCol_loss, total_iters)
                    log.scalar_summary('pos_loss_batch', spPos_loss, total_iters)

                if total_iters % args.recFre == 0:
                    write_log(args, total_train_loss / len(TrainImgLoader), total_iters,
                                       viz, imgL_crop, imgR_crop, disp_crop_L,log, num_log_img)

                total_iters += 1
                start_time = time.time()


           log.scalar_summary('avg_loss', total_train_loss / len(TrainImgLoader), epoch)
           print('full training time = %.2f HR/epoch' %((time.time() - start_full_time)/3600))
           torch.cuda.empty_cache()

        if not args.test_only:
            torch.cuda.synchronize()
            savefilename = args.savemodel + '/' + saveName
            b_best = False
            save_ckpt({
                'start_epoch': epoch + 1,
                'best_err': best_err,
                'state_dict': model.state_dict(),
                'optimz_state': optimizer.state_dict()
            }, savefilename, epoch, b_best)

            list_ckpt = glob(os.path.join(savefilename, 'epoch_*.tar'))
            list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            if len(list_ckpt) > 5:
                os.remove(list_ckpt[0])


        # validate every 2 epochs to save time
        if epoch % 2 == 0 and epoch < args.epochs:
            continue

        # ------------- TEST ------------------------------------------------------------
        total_test_loss = 0

        test_iter = len(TestImgLoader)
        for val_batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            with torch.no_grad():
                test_loss, viz = test(imgL,imgR, disp_L)
                print('Epoch [%.2d: %.4d] test loss = %.3f' %(epoch, val_batch_idx, test_loss))
                total_test_loss += test_loss

            # for the last epoch, evaluate all test data
            # if (not args.test_only) and val_batch_idx >= val_epoch_size-1 and epoch<args.epochs:
            #    test_iter = min(len(TestImgLoader), val_epoch_size)
            #    # print("break {}, {}, {}".format(args.test_only, val_batch_idx, val_epoch_size))
            #    break

        print('total test loss = %.3f' % (total_test_loss /test_iter) )
        if not args.test_only:
            b_best = best_err > total_test_loss / test_iter
            best_err = min(total_test_loss / test_iter, best_err)
            if b_best:
                shutil.copyfile(savefilename + '/epoch_%d.tar' % epoch, savefilename + '/best_model.tar')

            write_log(args, total_test_loss/test_iter, epoch,
                               viz, imgL, imgR,  disp_L, val_log, num_log_img, b_train=False)
            torch.cuda.empty_cache()

        #----------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
    
