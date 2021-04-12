from utils import logger
import shutil
import numpy as np
import torch
from skimage.segmentation import mark_boundaries
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.sp_utils import *
from math import ceil
import sys

def write_log(args, loss, step, viz, imgL, imgR, disp_L, logger, num_log_img, b_train=True, thres=5, b_spixelOnly=False):
    with torch.no_grad():
        sz_list = args.sz_list #[16, 8, 16, 32, 64]
        mean_values = torch.tensor([0.485, 0.456, 0.406], dtype=imgL.dtype).view(3, 1, 1).to(imgL.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=imgL.dtype).view(3, 1, 1).to(imgL.device)

        num_log_img = min(num_log_img, imgL.shape[0])

        if b_train:
            pass
            # logger.image_summary('output2', viz['output2'][0:num_log_img], step)
            # logger.image_summary('output1', viz['output1'][0:num_log_img], step)
            # logger.image_summary('disturb_img', viz['disturb_img'][0:num_log_img], step)
            # padH, padW = 0, 0
        else:
            logger.scalar_summary('avg_loss', loss, step)
            # padH, padW = args.val_img_height-args.real_img_height, 0

        ori_imgL, ori_imgR = (imgL[0:num_log_img,:, :, :] * std + mean_values), (imgR[0:num_log_img,:,:,:] * std + mean_values)
        # sp_imgL, sp_imgR = (viz['spImg_l'][0:num_log_img, :, :, :] * std + mean_values), ( viz['spImg_r'][0:num_log_img, :, :, :] * std + mean_values)
        logger.image_summary('imgL', ori_imgL[:, :, :, :], step)
        logger.image_summary('imgR', ori_imgR[:, :, :, :], step)

        if not b_spixelOnly:
            vis_err = np.abs(viz['final_output'] - disp_L.cpu().numpy()) * viz['mask']
            logger.image_summary('gt0', disp_L[0:num_log_img], step)
            logger.image_summary('final_res', viz['final_output'][0:num_log_img], step)
            logger.image_summary('err', val2uint8(vis_err[0:num_log_img], thres), step)
            logger.histo_summary('disparity_hist', viz['final_output'], step)
            logger.histo_summary('gt_hist', np.asarray(disp_L), step)


        # logger.image_summary('spImgL', sp_imgL, step)
        # logger.image_summary('spImgR', sp_imgR, step)


        # spixel viz
        # print(ori_imgL.shape, viz["spixel_idx_list"][0].shape, viz['sp_assign_L'][0].shape)
        # print(ori_imgR.shape, viz["spixel_idx_list"][1].shape, viz['sp_assign_L'][1].shape)


        spMap_list_L, prob_map_stack_L = update_spixl_map( viz["spixel_idx_list"], viz['sp_assign_L'])
        # spMap_list_R, prob_map_stack_R = update_spixl_map( viz["spixel_idx_list"], viz['sp_assign_R'])
        #
        imgL_save = make_grid(ori_imgL, nrow=num_log_img,padding=0)
        # imgR_save = make_grid(ori_imgR, nrow=num_log_img,padding=0)
        #
        #
        for i in range(len(spMap_list_L)):

            spixel_lab_save_L = make_grid(spMap_list_L[i][0:num_log_img, :, : ,:], nrow=num_log_img,padding=0)[0,:,:]
            # spixel_lab_save_R = make_grid(spMap_list_R[i][0:num_log_img, :, : ,:], nrow=num_log_img,padding=0)[0,:,:]

            # print(imgL_save.shape, spixel_lab_save_L.shape)

            # if i > 0 and not b_train:
            #     _, _, h, w =  spMap_list_L[i].shape
            #     pad_ori_imgL = F.pad(ori_imgL[:,:,::4,::4], (int(w-args.val_img_width/4), 0, int(h-args.val_img_height/4),0), "constant", 0)
            #     pad_ori_imgR = F.pad(ori_imgR[:, :, ::4, ::4], (int(w - args.val_img_width/4), 0, int(h - args.val_img_height/4), 0),"constant", 0)
            #     imgL_save = make_grid(pad_ori_imgL, nrow=num_log_img, padding=0)
            #     imgR_save = make_grid(pad_ori_imgR, nrow=num_log_img, padding=0)
            #
            #     spixel_viz_L, _ = get_spixel_image(args, imgL_save, spixel_lab_save_L, b_pad=True)
            #     spixel_viz_R, _ = get_spixel_image(args, imgR_save, spixel_lab_save_R, b_pad=True)
            # else:
            spixel_viz_L, _ = get_spixel_image(args, imgL_save, spixel_lab_save_L)
            # spixel_viz_R, _ = get_spixel_image(args, imgR_save, spixel_lab_save_R)

            tmpW = spixel_viz_L.shape[-1]//num_log_img
            spixel_L = np.zeros((num_log_img, 3, spixel_viz_L.shape[-2], tmpW))
            # spixel_R = np.zeros((num_log_img, 3, spixel_viz_L.shape[-2], tmpW))
            for k in range(num_log_img):
                spixel_L[k] = spixel_viz_L[:, :, k * tmpW :(k + 1) * tmpW]
                # spixel_R[k] = spixel_viz_R[:, :, k * tmpW :(k + 1) * tmpW]


            logger.image_summary('spixel_l_%d_%d'%(sz_list[i], i), np.ascontiguousarray(spixel_L), step)
            # logger.image_summary('spixel_r_%d_%d' %( sz_list[i], i), np.ascontiguousarray(spixel_R), step)
            # logger.image_summary('prob_map_l_%d_%d' % (sz_list[i], i),prob_map_stack_L[i][0:num_log_img].detach().cpu().numpy(), step)
            # logger.image_summary('prob_map_r_%d_%d' % (sz_list[i], i), prob_map_stack_R[i][0:num_log_img].detach().cpu().numpy(), step)

        # recover disparity error
        # _, fh, fw = disp_L.shape
        # sz = 4
        # tgt_h, tgt_w = ceil(float(fh) / sz) * sz, ceil(float(fw) / sz) * sz
        # pad_h, pad_w = tgt_h - fh, tgt_w - fw
        # match_h = viz['sp_assign_L'][0].shape[-2] - fh #test time will does not match 540 vs 544
        # pad_disp = F.pad(disp_L.unsqueeze(1).to(viz['sp_assign_L'][0].device), (pad_w, 0, pad_h, 0), "constant",0)
        # pad_mask = F.pad(viz['sp_assign_L'][0][:,:,match_h:,:], (pad_w, 0, pad_h, 0), "constant", 0)
        # # print (pad_disp.shape,pad_mask.shape)
        # rec_disp =  upfeat(poolfeat(pad_disp, pad_mask,sz, sz),
        #                    pad_mask, sz, sz).squeeze(1).detach().to(disp_L.device)
        # viz["rec_err"] = torch.abs(rec_disp[:,pad_h:, pad_w:] - disp_L).cpu().numpy() * viz['mask']
        # logger.image_summary('rec_disp_err', val2uint8(viz['rec_err'][0:num_log_img], thres), step)

def get_spixel_image(args, given_img, spix_index, b_pad = False, b_enforce_connect = False):

    if not isinstance(given_img, np.ndarray):
        given_img_np_ = given_img.detach().cpu().numpy().transpose(1,2,0)
    else: # for cvt lab to rgb case
        given_img_np_ = given_img

    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy().transpose(0,1)
    else:
        spix_index_np = spix_index

    if b_pad:
        # h, w = spix_index_np.shape
        given_img_np = given_img_np_# [ ::4, ::4,:] #h, w, c
        # given_img_np = np.pad(given_img_np,((int(h-args.val_img_height/4),0), (0, 0), (0, 0)), mode='constant')
    else:
        h, w = spix_index_np.shape
        given_img_np = cv2.resize(given_img_np_, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    # print (spix_index_np.shape, given_img_np.shape)
    # if b_enforce_connect:
    #     spix_index_np = spix_index_np.astype(np.int64)
    #     segment_size = (given_img_np_.shape[0] * given_img_np_.shape[1]) / (int(n_spixels) * 1.0)
    #     min_size = int(0.06 * segment_size)
    #     max_size =  int(3 * segment_size)
    #     spix_index_np = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]

    spixel_bd_image = mark_boundaries(given_img_np/np.max(given_img_np), spix_index_np.astype(int), color = (0,1,1)) #cyna 1 ,0, 0
    return spixel_bd_image.astype(np.float32).transpose(2,0,1), spix_index_np #


def save_ckpt(state, save_path, epoch, b_best):
    torch.save(state, save_path + '/epoch_%d.tar'%epoch)
    if b_best:
        shutil.copyfile(save_path + '/epoch_%d.tar'%epoch, save_path + '/best_model.tar')


def val2uint8(mat,maxVal, minVal=0):
    maxVal_mat = np.ones(mat.shape) * maxVal
    minVal_mat = np.ones(mat.shape) * minVal

    mat_vis = np.where(mat > maxVal_mat, maxVal_mat, mat)
    mat_vis = np.where(mat < minVal_mat, minVal_mat, mat_vis)
    return (mat_vis * 255. / maxVal).astype(np.uint8)


def update_spixl_map (spixl_map_idx_in, assig_map_in):

    new_spixl_map_stack = []
    prob_map_stack = []
    for i, assig_map in enumerate(assig_map_in):
        b,_,h,w = assig_map.shape
        spixl_map_idx = spixl_map_idx_in[i]
        spixl_map_idx = spixl_map_idx.to(assig_map.device)
        # assig_map = assig_map.to(spixl_map_idx.device)
        # print(spixl_map_idx.shape, assig_map.shape)
        # if isinstance(spixl_map_idx_in, list):  # spixl idx tensor list
        #     spixl_map_idx = spixl_map_idx_in[i]
        # else:                                   # spixl idx tensor
        #     spixl_map_idx = F.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')

        assig_max, prob_idx = torch.max(assig_map, dim=1, keepdim= True)
        assignment_ = torch.where(assig_map == assig_max, torch.ones(assig_map.shape).to(assig_map.device),torch.zeros(assig_map.shape).to(assig_map.device))
        new_spixl_map_ = spixl_map_idx * assignment_ # winner take all
        new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)
        new_spixl_map_stack.append(new_spixl_map)

        # prob_map_save = assign2uint8(prob_idx)
        # prob_map_stack.append(prob_map_save)

    return new_spixl_map_stack, prob_map_stack


def assign2uint8(assign):
    #red up, green mid, blue down
    # print(assign.shape)
    b,c,h,w = assign.shape

    red = torch.cat([torch.ones(size=assign.shape),  torch.zeros(size=[b,2,h,w])],dim=1).to(assign.device)

    green = torch.cat([ torch.zeros(size=[b,1,h,w]),
                      torch.ones(size=assign.shape),
                      torch.zeros(size=[b,1,h,w])],dim=1).to(assign.device)

    blue  = torch.cat([torch.zeros(size=[b,2,h,w]),
                       torch.ones(size=assign.shape)],dim=1).to(assign.device)

    black = torch.zeros(size=[b,3,h,w]).to(assign.device)
    white = torch.ones(size=[b,3,h,w]).to(assign.device)
    # up probablity
    mat_vis = torch.where(assign.type(torch.float) < 0. , white, black)
    mat_vis = torch.where(assign.type(torch.float) >= 0. , red* (assign.type(torch.float)+1)/3, mat_vis)
    mat_vis = torch.where(assign.type(torch.float) >= 3., green*(assign.type(torch.float)-2)/3, mat_vis)
    mat_vis = torch.where(assign.type(torch.float) >= 6., blue * (assign.type(torch.float) - 5.) / 3, mat_vis)

    return (mat_vis * 255.).type(torch.uint8)


'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)
