from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule_pre import *
from models.spixel_module_old_small import SpixelNet
from utils import *
from math import ceil

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, args, spixel_ckpts, b_finetune=False):
        super(PSMNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # sp utils
        self.spixel_size_list = args.sz_list#[16,8,16,32,64]
        self.maxdisp = args.maxdisp
        _, self.spix_idx_list,  _, self.XY_feat_full,  _=\
            init_spixel_grid(args, self.spixel_size_list)

        _, self.val_spix_idx_list, _, _, _ = \
            init_spixel_grid(args, self.spixel_size_list,b_train=False)

        self.mean_values = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # self.divisor = 1

        self.rgb2Lab = rgb2Lab_torch
        self.build_labxy_feat = build_LABXY_feat
        self.poolfeat = poolfeat
        self.upfeat = upfeat

        # stereo matching
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # with torch.no_grad():
        self.spixel4 = SpixelNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # print(spixel_ckpts)
        # if not b_finetune:
        #     self.spixel4.load_state_dict(torch.load(spixel_ckpts)['state_dict'])

    def forward(self, left, right, disp_true=None, b_joint=False):
        # ======== Feat Extract ========
        b = left.shape[0]
        sz = self.spixel_size_list[0]

        if self.training and b_joint:
            maskL_full_4, _ = self.spixel4(left)
            maskR_full_4, _ = self.spixel4(right)

            spImg_l = self.poolfeat(left, maskL_full_4, sz, sz)
            spImg_r = self.poolfeat(right, maskR_full_4, sz, sz)
        else:
            with torch.no_grad():
                maskL_full_4, _ = self.spixel4(left)
                maskR_full_4, _ = self.spixel4(right)
                spImg_l = self.poolfeat(left, maskL_full_4, sz, sz)
                spImg_r = self.poolfeat(right, maskR_full_4, sz, sz)

        refimg_fea  = self.feature_extraction(spImg_l)
        targetimg_fea = self.feature_extraction(spImg_r)

        if sz == 4:
            self.divisor = 1
            self.regress_disp = int(self.maxdisp)
        elif sz == 16:
            # XY_feat = self.XY_feat_quater
            self.regress_disp = int(self.maxdisp / 4)
            self.divisor = 4


        col_err = None
        pos_err = None
        if self.training and b_joint:
            with torch.no_grad():
                XY_feat = self.XY_feat_full.to(spImg_l.device)
                imgL_lab = self.rgb2Lab(left, self.mean_values, self.std)
                imgR_lab = self.rgb2Lab(right, self.mean_values, self.std)
                LABXY_featL_full = self.build_labxy_feat(imgL_lab, XY_feat, b, self.device)
                LABXY_featR_full = self.build_labxy_feat(imgR_lab, XY_feat, b, self.device)

            pad_labxy_sck = torch.cat([LABXY_featL_full, LABXY_featR_full], dim=0)
            pad_mask_sck = torch.cat([maskL_full_4, maskR_full_4], dim=0)
            pooled_labxy = self.poolfeat(pad_labxy_sck, pad_mask_sck, sz, sz)
            reconstr_fea = self.upfeat(pooled_labxy, pad_mask_sck, sz, sz)

            loss_map_L = (reconstr_fea - pad_labxy_sck)  # [:, :, pad_h:, pad_w:])
            # probably we used this one in our paper result
            # col_err =  b * torch.norm(loss_map_L[:, :3, :, :], p=2, dim=1).mean().unsqueeze(0)
            # pos_err =  b * torch.norm(loss_map_L[:, -2:, :, :], p=2, dim=1).mean().unsqueeze(0)

            # the correct one w.r.t. the paper should be this one
            col_err =   b * torch.norm(loss_map_L[:b, :3, :, :], p=2, dim=1).mean().unsqueeze(0) + \
                        b * torch.norm(loss_map_L[b:, :3, :, :], p=2, dim=1).mean().unsqueeze(0)
            pos_err =   b * torch.norm(loss_map_L[:b, -2:, :, :], p=2, dim=1).mean().unsqueeze(0) + \
                        b * torch.norm(loss_map_L[b:, -2:, :, :], p=2, dim=1).mean().unsqueeze(0)
        # ========== disparity ============
        # torch.backends.cudnn.benchmark = False
        #matching
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], int(refimg_fea.size()[1] * 2), self.maxdisp // sz,
                                               refimg_fea.size()[2], refimg_fea.size()[3]).fill_(0.))

        for i in range(self.maxdisp // sz):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0

        # the cost is for spixel center disparity
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2


        if self.training:
            # cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)
            # cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            cost2 = torch.squeeze(cost2, 1)
            # cost1 = self.upDisp(cost1)
            if sz == 4:
                cost1 = self.upfeat(cost1, maskL_full_4, sz, sz)
                cost1 = F.upsample(cost1.unsqueeze(1), [self.maxdisp, cost1.size()[2], cost1.size()[3]], mode='trilinear',align_corners=True).squeeze(1)
                pred1 = F.softmax(cost1,dim=1)
                pred1 = disparityregression(self.maxdisp, self.divisor)(pred1)


                cost2 = self.upfeat(cost2, maskL_full_4, sz, sz)
                cost2 = F.upsample(cost2.unsqueeze(1), [self.maxdisp, cost2.size()[2], cost2.size()[3]], mode='trilinear', align_corners=True).squeeze(1)
                pred2 = F.softmax(cost2,dim=1)
                pred2 = disparityregression(self.maxdisp, self.divisor)(pred2)
            else:
                cost1 = F.upsample(cost1.unsqueeze(1), [self.regress_disp, cost1.size()[2], cost1.size()[3]],
                                   mode='trilinear', align_corners=True).squeeze(1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = disparityregression(self.regress_disp, self.divisor)(pred1)
                pred1 = self.upfeat(pred1.unsqueeze(1), maskL_full_4, sz, sz).squeeze(1)

                cost2 = F.upsample(cost2.unsqueeze(1), [self.regress_disp, cost2.size()[2], cost2.size()[3]],
                                   mode='trilinear', align_corners=True).squeeze(1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = disparityregression(self.regress_disp, self.divisor)(pred2)
                pred2 = self.upfeat(pred2.unsqueeze(1), maskL_full_4, sz, sz).squeeze(1)

        # cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        # cost3 = self.upDisp(cost3)
        if sz == 4:
            cost3 = self.upfeat(cost3, maskL_full_4, sz, sz)
            cost3 = F.upsample(cost3.unsqueeze(1), [self.maxdisp, cost3.size()[2], cost3.size()[3]], mode='trilinear', align_corners=True).squeeze(1)
            pred3 = F.softmax(cost3,dim=1)
            #For your information: This formulation 'softmax(c)' learned "similarity"
            #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
            #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
            pred3 = disparityregression(self.maxdisp, self.divisor)(pred3)
        else:

            cost3 = F.upsample(cost3.unsqueeze(1), [self.regress_disp, cost3.size()[2], cost3.size()[3]],
                               mode='trilinear', align_corners=True).squeeze(1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparityregression(self.regress_disp, self.divisor)(pred3)
            pred3 = self.upfeat(pred3.unsqueeze(1), maskL_full_4, sz, sz).squeeze(1)

        # in multi-GPU case pytorch can combine the batch dim even if I use list as return
        if self.training:
            # compute loss here
            mask = (disp_true > 0) & (disp_true < self.maxdisp)
            mask.detach_()

            err1 = b * F.smooth_l1_loss(pred1[mask], disp_true[mask], size_average=True).unsqueeze(0)
            err2 = b * F.smooth_l1_loss(pred2[mask], disp_true[mask], size_average=True).unsqueeze(0)
            err3 = b * F.smooth_l1_loss(pred3[mask], disp_true[mask], size_average=True).unsqueeze(0)

            return (err1, None),  (err2, None), (err3, pred3.detach()) ,  \
                   (col_err, pos_err),   None,   maskL_full_4,  None, self.spix_idx_list, \
                   None, mask.type(torch.float)
            # pred1, pred2, pred3,  loss_map_L,   None,   maskL,  maskR, self.spix_idx_list, (spImg_l, spImg_r)#, disturb_img
        else:

            return pred3,  maskL_full_4,  None, self.val_spix_idx_list#, (spImg_l, spImg_r)
