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
import shutil
from utils import *

from glob import glob
from dataloader import listfiles as ls
from dataloader import MiddleburyLoader as DA
# from utils import logger, save_res

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=768,
                    help='maxium disparity')
parser.add_argument('--model', default='prePSMNet_small',
                    help='select model')
parser.add_argument('--datapath', default='/home/fuy34/stereo_data/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=4, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--test_batchsize', type=int, default=4, #https://github.com/JiaRenChang/PSMNet/issues/73
                    help='number of epochs to train(default:12 or 8)')
parser.add_argument('--preTrain_spixel', default= './preTrain_spixel_old/',
                    help='preTrain model')

parser.add_argument('--loadmodel', default=#'/data/Fengting/stereo_training/SPPSMNet/useful_sceneflow/joint_sp4_spW0.1_mW15_ep13_b8/best_model.tar',
'/data/Fengting/stereo_training/SPPSMNet/useful_sceneflow/jointTrain_sp4only_spw0.1_posw15/best_model.tar',
                    help='load model')
parser.add_argument('--logname', default='my_finetue_tst',
                    help='log name')
parser.add_argument('--savemodel', default='/data/Fengting/stereo_training/SPPSMNet/useful_HRVS/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


parser.add_argument('--m_w', type=float, default=30,
                    help='slic position weight')
parser.add_argument('--sp_w', type=float, default=0.1,
                    help='spixel loss weight')
parser.add_argument('--sz_list', type=float, default= [16], #, 8, 16
                    help='spixel loss weight')

parser.add_argument('--train_img_height', '-t_imgH', default=256*4, #384,
                    type=int,  help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default= 512*4, #768,
                    type=int, help='img width')

parser.add_argument('--val_img_height', '-v_imgH', default=2048, #512 368  2048, 1792
                    type=int,  help='img height_must be 16*n') #
parser.add_argument('--val_img_width', '-v_imgW', default= 2432,   #960 1232 3072, 2240
                    type=int, help='img width must be 16*n')


parser.add_argument('--epoch_size', default=1e5,   #960
                    type=int, help='img width must be 16*n')

parser.add_argument('--continue_train', action='store_true', default=False,
                    help='continue training')
parser.add_argument('--recFre', type=int, default=4,
                    help='recording training status frequence (epoch)')
parser.add_argument('--saveFre', type=int, default=10,
                    help='recording checkpoint frequence (epoch)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


left_img_hr, right_img_hr, left_disp_hr,test_left_img, test_right_img, test_left_disp= ls.dataloader( args.datapath+'/HR_VS/carla-highres/trainingF/', typ='trainval', b_HRonly=True)
all_left_img  = left_img_hr
all_right_img = right_img_hr
all_left_disp =  left_disp_hr

train_epoch_size = min(args.epoch_size* args.batchsize, len(all_left_img)) #only for debug
val_epoch_size = min(args.epoch_size * args.batchsize, len(test_left_disp))


TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img[:train_epoch_size],all_right_img[:train_epoch_size],all_left_disp[:train_epoch_size], True,
                          hw_list=[args.train_img_height,args.train_img_width,args.val_img_height,args.val_img_width]),
         batch_size= args.batchsize, shuffle= True, num_workers= 4, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img[:val_epoch_size],test_right_img[:val_epoch_size],test_left_disp[:val_epoch_size], False,
                          hw_list=[args.train_img_height,args.train_img_width,args.val_img_height,args.val_img_width]),
         batch_size= args.test_batchsize, shuffle= False, num_workers= 4, drop_last=True)


if args.model == 'prePSMNet':
    # model = prePSMNet(args, None, b_finetune= True)
    pass
elif args.model == 'prePSMNet_small':
    model = prePSMNet_small(args, None, b_finetune=True)
else:
    print('no model')

if args.cuda:
    torch.backends.cudnn.benchmark = True
    model = nn.DataParallel(model)
    model.cuda()


train_params =  [param for name, param in model.named_parameters() if param.requires_grad and not 'spixel' in name]
spixel_params = [param for name, param in model.named_parameters() if param.requires_grad and 'spixel' in name ] #and 'spixel4' not in name

optimizer = optim.Adam( [
                        {'params': train_params},
                        {'params': spixel_params, 'weight_decay': 4e-4, 'lr':0.0001}
                        ], lr=0.001, betas=(0.9, 0.999)) #{'params': [s1, s2]}, {'params': spixel_params, 'weight_decay': 4e-4}

# optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
spixel_ckpts = args.preTrain_spixel+'/smallNet_16.tar'

if args.loadmodel is not None:
    if args.continue_train:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        start_epoch = state_dict['start_epoch']
        optimizer.load_state_dict(state_dict['optimz_state'])
        best_err = state_dict['best_err']
        max_epo = state_dict['max_epo']
        total_iters = (start_epoch-1) * len(TrainImgLoader)
    else:
        state_dict = torch.load(args.loadmodel)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict['state_dict'].items() if 'spixel4' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.module.spixel4.load_state_dict(torch.load(spixel_ckpts)['state_dict'])

        start_epoch = 1
        best_err = 1e4
        total_iters = 0
        max_acc, max_epo = 0, 0

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        # mask = (disp_true > 0) & (disp_true < args.maxdisp)
        # mask.detach_()
        #----

        optimizer.zero_grad()

        outputs = model(imgL,imgR, disp_L, b_joint=True)

        output1 = outputs[0][0]  # torch.squeeze(outputs[0], 1)
        output2 = outputs[1][0]  # torch.squeeze(outputs[1], 1)
        output3 = outputs[2][0]  # torch.squeeze(outputs[2], 1)
        # disp_loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
        disp_loss = (0.5 * output1.sum() + 0.7 * output2.sum() + output3.sum())/args.batchsize

        loss_col = 0
        loss_pos = 0
        errs = outputs[3]
        # loss_map_R = outputs[4]

        for i in range(len(args.sz_list)):  # todo 5
            a = errs[0].sum() / args.batchsize
            b = args.m_w /(args.sz_list[i]) * errs[1].sum() / args.batchsize # math.sqrt
            loss_col += a
            loss_pos += b
            print("level {}, loss_col: {} loss_pos: {}".format(i, args.sp_w * a, args.sp_w * b))

        spixle_loss = args.sp_w * (loss_col + loss_pos)
        loss = disp_loss  + spixle_loss


        viz = {}
        viz['final_output'] = outputs[2][1].detach().cpu().numpy()
        viz['sp_assign_L'] = [outputs[5]]
        viz["spixel_idx_list"] = outputs[7]
        viz['mask'] = outputs[-1].cpu().numpy()

        loss.backward()
        optimizer.step()

        return loss.item(), disp_loss.item(), args.sp_w * loss_col.item(), args.sp_w * loss_pos.item(), viz #0, 0, viz #

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        with torch.no_grad():
            outputs = model(imgL,imgR)
            mask = (disp_true > 0)


        output3 = torch.squeeze(outputs[0], 1) #
        pred_disp = output3#.data.cpu()

        viz = {}
        viz['final_output'] = output3.detach().cpu().numpy()
        viz['mask'] = (disp_true>0).type(torch.float).detach().cpu().numpy()
        viz['sp_assign_L'] = [outputs[1]]
        # viz['sp_assign_R'] = outputs[2]
        viz["spixel_idx_list"] = outputs[3]

        #computing 3-px error#
        if len(disp_true[mask]) == 0:
            loss = 0
        else:
            loss = torch.sum(torch.abs(pred_disp[mask] - disp_true[mask])) / torch.sum(
                mask.type(torch.float))  # end-point-error

        return loss, viz

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 150:
       lr = 0.001
    else:
       lr = 0.0001
       for param_group in optimizer.param_groups:
           param_group['lr'] = lr
    print(lr)

    return lr

def main():
    global start_epoch, total_iters, best_err, max_epo
    num_log_img = 4
    # train_epoch_max = 1e5
    saveName = args.logname + "_spW{}_mW{}_ep{}_b{}".format(args.sp_w, args.m_w, args.epochs, args.batchsize)
    log = logger.Logger(args.savemodel, name=saveName , s_train = 'train')
    val_log = logger.Logger(args.savemodel, name=saveName , s_train = 'val')

    start_full_time = time.time()

    for epoch in range(start_epoch, args.epochs+1):
       print('This is %d-th epoch, total iter: %d' % (epoch, len(TrainImgLoader)))
       total_train_loss = 0
       total_test_loss = 0
       lr = adjust_learning_rate(optimizer,epoch)
       log.scalar_summary('lr', lr, epoch)

        ## training ##
       start_time = time.time()
       for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

           torch.cuda.synchronize()
           data_time = time.time() - start_time

           loss, disp_loss, spCol_loss, spPos_loss, viz = train(imgL_crop,imgR_crop, disp_crop_L)
           torch.cuda.synchronize()
           print('Epoch [%.2d: %.4d] training loss = %.3f , data time = %.2f, total time = %.2f' %(epoch, batch_idx, loss, data_time,  time.time() - start_time))
           total_train_loss += loss

           if total_iters % 10 == 0:
               log.scalar_summary('total_loss_batch', loss, total_iters)
               log.scalar_summary('disp_loss_batch', disp_loss, total_iters)
               log.scalar_summary('spixel_loss_batch', spCol_loss + spPos_loss, total_iters)
               log.scalar_summary('col_loss_batch', spCol_loss, total_iters)
               log.scalar_summary('pos_loss_batch', spPos_loss, total_iters)

           total_iters += 1
           start_time = time.time()


       print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
       if epoch % args.recFre == 0:
           with torch.no_grad():
                save_res.write_log(args, total_train_loss / len(TrainImgLoader), epoch,
                              viz, imgL_crop, imgR_crop, disp_crop_L, log, num_log_img,thres=15)
       else:
           log.scalar_summary('avg_loss', total_train_loss / len(TrainImgLoader), epoch)

       ## Test ##
       # val_epoch_size = 1e4
       test_iter = len(TestImgLoader)
       for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
           test_loss, viz = test(imgL,imgR, disp_L.clone())
           print('Iter %d  error in val = %.3f' %(batch_idx, test_loss))
           total_test_loss += test_loss

       print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/test_iter))

        # torch.cuda.synchronize()
       if epoch % args.saveFre == 0:
           savefilename = args.savemodel + '/' + saveName

           # b_best = total_test_loss/len(TestImgLoader)*100 < best_err
           save_res.save_ckpt({
                'start_epoch': epoch + 1,
                'best_err': best_err,
                'max_epo': max_epo,
                'state_dict': model.state_dict(),
                'optimz_state': optimizer.state_dict()
            }, savefilename, epoch, False)

           list_ckpt = glob(os.path.join(savefilename, 'epoch_*.tar'))
           list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
           if len(list_ckpt) > 5:
               os.remove(list_ckpt[0])

       if total_test_loss / test_iter < best_err:
           best_err = total_test_loss / test_iter
           max_epo = epoch
           savefilename = args.savemodel + '/' + saveName
           # shutil.copyfile(savefilename + '/epoch_%d.tar' % epoch, savefilename + '/best_model.tar')
           torch.save({
               'start_epoch': epoch + 1,
               'best_err': best_err,
               'max_epo': max_epo,
               'state_dict': model.state_dict(),
               'optimz_state': optimizer.state_dict()
           }, savefilename + '/best_model.tar')

           print('Best epoch %d ; Best test error = %.3f' % (max_epo, best_err))


       if epoch % args.recFre == 0:
           with torch.no_grad():
               save_res.write_log(args, total_test_loss / test_iter, epoch,
                                  viz, imgL, imgR, disp_L, val_log, num_log_img, b_train=False,thres=15)
       else:
           val_log.scalar_summary('avg_loss', total_test_loss / test_iter, epoch)

    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print('Best epoch: {}\t Best_err: {}\t'.format(max_epo, best_err))


if __name__ == '__main__':
   main()
