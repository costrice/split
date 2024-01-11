# -*- coding: utf-8 -*-

import os
import re
import time
import torchvision

import scipy.io as sio
import numpy as np
from skimage import io
import pywavefront
from tqdm import tqdm

import config
import yaml
from pdb import set_trace as st
import torch

import torchvision.transforms as T
import cv2

from utils.general import *
import pandas as pd

def draw_outputs_full(pred, input, pred_render=None, gt_render=None, vis_num=6, vis_size=128, visualize_test=True, face_hdr=True):
    """
    Make num_vis of data visualize the output of network alongside the label.
    Args:
        pred: the prediction or output of networks.
        input: dict[..., 'envmap'] of the input and label of the corresponding batch.
        pred_render: dict['Diffuse', 'Specular']: optional, the predicted rendered lobe.
        gt_render: dict['Diffuse', 'Specular']: optional, the ground-truth rendered lobe.
        vis_num: the number of data to visualize.
    Returns:
        A visualization in the type of torch.Tensor.
    """
    padding_size = 0
    if 'name' in input.keys():
        input.pop("name")
    caption = ""
    
    batch_size = pred.shape[0]
    # preprocess
    ## detach and cpu
    pred = pred.cpu().detach()
    input = {k: v.cpu().detach() for k, v in input.items()}
    if pred_render is not None:
        pred_render = {k: v.cpu().detach() for k, v in pred_render.items()}
    if gt_render is not None:
        gt_render = {k: v.cpu().detach() for k, v in gt_render.items()}

    ## get the first vis_num of data
    assert batch_size >= vis_num

    pred = pred[:vis_num]
    input = {k: v[:vis_num] for k, v in input.items()}
    if pred_render is not None:
        pred_render = {k: v[:vis_num] for k, v in pred_render.items()}
    if gt_render is not None:
        gt_render = {k: v[:vis_num] for k, v in gt_render.items()}

    ## normal = (normal + 1) * 0.5 for visualization
    if 'normal' in input.keys():
        if 'normal_face' in input.keys():
            input['normal'] = (input['normal'] + 1) * 0.5
            sp_mask = generate_sphere_mask(input['normal'].shape[3])
            input['normal'] = torch.mul(input['normal'], sp_mask)
        else:
            input['normal'] = (input['normal'] + 1) * 0.5
            input['normal'] = torch.mul(input['normal'], input['mask'])
    
    if 'normal_face' in input.keys():
        input['normal_face'] = (input['normal_face'] + 1) * 0.5
        input['normal_face'] = torch.mul(input['normal_face'], input['mask_face'])
        # st()

    if 'mask' in input.keys():
        input.pop("mask")
    if 'mask_face' in input.keys():
        input.pop("mask_face")
    
    sp_mask = generate_sphere_mask(input['envmap'].shape[3])
    pred = torch.mul(pred, sp_mask)
    for vis_idx in range(vis_num):
        pred[vis_idx] = map_hdr_clip(pred[vis_idx])
        for key in input.keys():
            if key == 'normal' or key == 'normal_face':
                input[key][vis_idx] = input[key][vis_idx]
            elif key == 'envmap':
                input[key][vis_idx] = map_hdr_clip(input[key][vis_idx])
            elif key == 'face_face':
                if face_hdr:
                    input[key][vis_idx] = map_hdr_clip(input[key][vis_idx])
                else:
                    input[key][vis_idx] = input[key][vis_idx]
            else: 
                input[key][vis_idx] = map_hdr_clip(input[key][vis_idx])
        if pred_render is not None:
            for key in pred_render.keys():
                pred_render[key][vis_idx] = map_hdr_clip(pred_render[key][vis_idx])
        if gt_render is not None:
            for key in gt_render.keys():
                gt_render[key][vis_idx] = map_hdr_clip(gt_render[key][vis_idx])
            
    # resize all the images to the same size
    meta_dict = {}
    for key in input.keys():
        input[key] = torchvision.transforms.functional.resize(input[key], vis_size)
        meta_dict['input_' + key] = input[key]
    pred = torchvision.transforms.functional.resize(pred, vis_size)
    meta_dict['pred'] = pred
    if pred_render is not None:
        for key in pred_render.keys():
            pred_render[key] = torchvision.transforms.functional.resize(pred_render[key], vis_size)
            meta_dict['pred_render_' + key] = pred_render[key]
    if gt_render is not None:
        for key in gt_render.keys():
            gt_render[key] = torchvision.transforms.functional.resize(gt_render[key], vis_size)
            meta_dict['gt_render_' + key] = gt_render[key]
    
    # concatenate all the images
    out_list = []
    pre_list = ['normal', 'specular', 'shading', 'envmap']

    if 'normal_face' in input.keys():
        # add face input images
        pre_list = ['face_face', 'normal_face', 'specular_face', 'shading_face'] + pre_list
    # add input images
    if set(input.keys()) == set(pre_list):
        for key in pre_list:
            if key != 'envmap':
                input[key] = torchvision.utils.make_grid(input[key], nrow=1, padding=padding_size)
                out_list.append(input[key])
                # grid_list.append(white_space)
                caption += key + " "
    else:
        if len(input.keys()) != 1:
            print('Not organized input images order.')
        for key in input.keys():
            if key != 'envmap':
                input[key] = torchvision.utils.make_grid(input[key], nrow=1, padding=padding_size)
                out_list.append(input[key])
                # grid_list.append(white_space)
                caption += key + " "

    # add predicted envmap
    pred = torchvision.utils.make_grid(pred, nrow=1, padding=0)
    out_list.append(pred)
    caption += "pred" + " "
    
    # add rendered pred images
    pre_list = ['specular', 'diffuse']
    if pred_render is not None:
        if set(pred_render.keys()) == set(pre_list):
            for key in pre_list:
                pred_render[key] = torchvision.utils.make_grid(pred_render[key], nrow=1, padding=padding_size)
                out_list.append(pred_render[key])
                # grid_list.append(white_space)
                caption += key + " "
        else:
            for key in pred_render.keys():
                pred_render[key] = torchvision.utils.make_grid(pred_render[key], nrow=1, padding=padding_size)
                out_list.append(pred_render[key])
                caption += key + " "

    # add ground-truth envmap
    input['envmap'] = torchvision.utils.make_grid(input['envmap'], nrow=1, padding=padding_size)
    out_list.append(input['envmap'])
    # grid_list.append(white_space)
    caption += 'envmap' + " "

    # add rendered gt images
    pre_list = ['specular', 'diffuse']
    if gt_render is not None:
        if set(gt_render.keys()) == set(pre_list):
            for key in pre_list:
                gt_render[key] = torchvision.utils.make_grid(gt_render[key], nrow=1, padding=padding_size)
                out_list.append(gt_render[key])
                # grid_list.append(white_space)
                caption += key + " "
        else:
            for key in gt_render.keys():
                gt_render[key] = torchvision.utils.make_grid(gt_render[key], nrow=1, padding=padding_size)
                out_list.append(gt_render[key])
                caption += key + " "

    grid_full = torch.cat(out_list, dim=2)
    grid_full = torch.clip(grid_full, 0, 1)
    # torchvision.utils.save_image(grid_full, './verbose/gf.png')
    # torchvision.utils.save_image(input['normal_face'], './verbose/nf.png')

    # save 
    # cv2.imwrite('./verbose/text.pdf', grid_full_np)
    # st()
    return grid_full, caption


def generate_sphere_mask_from_size_np(envmap):
    # Generate mask from envmap size
    # envmap: h x h x 3
    hw = envmap.shape[0]
    idx = np.arange(hw)
    idx_row = np.repeat(idx[:, None], repeats=hw, axis=1)
    idx_col = np.repeat(idx[None, :], repeats=hw, axis=0)
    mask = (((idx_row - hw / 2) ** 2 + (idx_col - hw / 2) ** 2) < (hw / 2) ** 2).astype(np.float32)
    # mask = torch.from_numpy(mask)
    # mask = mask.unsqueeze(0)
    # mask = mask.repeat(envmap.shape[0], 1, 1, 1)
    # mask = mask.astype(np.bool)
    mask = mask[:, :, None]
    return mask

def generate_sphere_mask_from_size(envmap):
    # Generate mask from envmap size
    hw = envmap.shape[2]
    idx = np.arange(hw)
    idx_row = np.repeat(idx[:, None], repeats=hw, axis=1)
    idx_col = np.repeat(idx[None, :], repeats=hw, axis=0)
    mask = (((idx_row - hw / 2) ** 2 + (idx_col - hw / 2) ** 2) < (hw / 2) ** 2).astype(np.float32)
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(envmap.shape[0], 1, 1, 1)
    mask = mask.type(torch.bool)
    return mask

def map_expname(net_type):
    map_dict = {'linet': 'face2env_reproduce', 
                'autoencoder': 'envmap_autoencoder', 
                'predictor': 'envmap_predictor'}
    return map_dict[net_type]

def generate_real_dict(gt):
    out_dict = {}
    smaller_edge = gt.shape[2]
    out_dict['fs'] = gt
    out_dict['ds2'] = T.Resize(size=int(smaller_edge / 2))(gt)
    out_dict['ds4'] = T.Resize(size=int(smaller_edge / 4))(gt)
    return out_dict

def generate_real_dict_google(gt):
    out_dict = {}
    smaller_edge = gt.shape[2]
    out_dict['fs'] = gt
    out_dict['ds2'] = T.Resize(size=int(smaller_edge / 2), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt)
    out_dict['ds4'] = T.Resize(size=int(smaller_edge / 4), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt)
    out_dict['ds8'] = T.Resize(size=int(smaller_edge / 8), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt)
    return out_dict

# def map_reinhard(hdr):
#     # print("Tonemaping using Reinhard's method ... ")
#     hdr = hdr / torch.max(torch.max(torch.max(hdr, dim=2, keepdim=True)[0], dim=1, keepdim=True)[0], dim=0, keepdim=True)[0]
#     # st()
#     tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
#     ldrReinhard = tonemapReinhard.process(hdr.numpy().transpose(1, 2,0))
#     # cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
#     # print("saved ldr-Reinhard.jpg")
#     ldrReinhard = ldrReinhard.transpose(2, 0, 1)
#     ldrReinhard = torch.from_numpy(ldrReinhard)
    
#     return ldrReinhard

def map_hdr(img):
    # size: (3, 16, 32)
    return (img / torch.max(torch.max(torch.max(img, dim=2, keepdim=True)[0],
                                      dim=1, keepdim=True)[0],
                            dim=0, keepdim=True)[0]) ** 0.454

def map_hdr_clip(img):
    # size: (3, 16, 32)
    g_img = img ** 0.454
    # brightness = 0.2126 * img[0, :, :] + 0.7152 * img[1, :, :] + 0.0722 * img[2, :, :]
    # g_img = (img * (0.3 / torch.mean(brightness))) ** 0.454
    return torch.clip(g_img, 0, 1)

def generate_sphere_mask(out_hw, device=None):

    """
    Generate sphere mask and normal of shape in Torch (c, out_hw, out_hw).
    """
    # generate row and column index
    idx = np.arange(out_hw)
    idx_row = np.repeat(idx[:, None], repeats=out_hw, axis=1)
    idx_col = np.repeat(idx[None, :], repeats=out_hw, axis=0)
    # generate each component of normal
    normal_x = (idx_col - out_hw / 2 + 0.5) / (out_hw / 2)
    normal_y = -(idx_row - out_hw / 2 + 0.5) / (out_hw / 2)
    normal_z = (np.maximum((1 - normal_x ** 2 - normal_y ** 2), 0)) ** 0.5
    # generate mask and normal
    mask_sp = (normal_x ** 2 + normal_y ** 2 <= 1)[:, :, None].astype(np.float32)
    # visualize
    # write_image(os.path.join('verbose', "mask_sp.png"), mask_sp)

    mask_torch = torch.from_numpy(mask_sp.transpose((2, 0, 1)))
    if device:
        mask_torch = mask_torch.to(device)
    return mask_torch


def save_test_inthewild(save_dir, test_img, test_loss, options, name_ls, num_test, exp_name='', faltten_save=False, generate_csv=True, start_idx=0):

    if test_img != None:
        img_path_dict = {}
        for key in test_img.keys():
            img_path_dict[key] = os.path.join(save_dir, 'imgs', key)
            if not os.path.exists(img_path_dict[key]):
                os.makedirs(img_path_dict[key])
        
        img_num = test_img['GT'].shape[0]

        hdr_list = ['GT', 'pred_hdr']
        sp_mask = generate_sphere_mask(test_img['GT'].shape[3])
        for idx in range(img_num):
            test_img['pred'][idx] = torch.mul(test_img['pred'][idx], sp_mask)
            test_img['pred_hdr'][idx] = torch.mul(test_img['pred_hdr'][idx], sp_mask)

        for key in test_img.keys():
            if key not in hdr_list:
                test_img[key] = map_hdr_clip(test_img[key])
            test_img[key] = test_img[key].cpu().detach().numpy()
            test_img[key] = np.transpose(test_img[key], (0, 2, 3, 1))
    
        for idx in range(img_num):
            for key in img_path_dict.keys():
                if key not in hdr_list and 'GT' not in key:
                    if key == 'pred':
                        write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx+start_idx:04d}_{key}.png"), test_img[key][idx])
                    elif key == 'face':
                        write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx+start_idx:04d}_{key}.png"), test_img[key][idx])
                    else:
                        write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx+start_idx:04d}_{key}.png"), srgb2linrgb(test_img[key][idx]))
                else:
                    if key == 'pred_hdr':
                        write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx+start_idx:04d}_{key}.hdr"), test_img[key][idx])
    else:
        print("test_img is None, skipping saving test images")


def save_test(save_dir, test_img, test_loss, options, name_ls, num_test, exp_name='', faltten_save=False, generate_csv=True, 
              save_loss_keys = None):

    # write config
    write_dict = {}
    write_dict['exp name'] = options.expname
    write_dict['test num'] = num_test
    write_dict['env name'] = 'all'
    
    
    if len(test_loss.keys()) != 0:
        if save_loss_keys is None:
            _key_ord = test_loss
        else:
            _key_ord = save_loss_keys
            
        for loss_term in _key_ord:
            if type(test_loss[loss_term]) == torch.Tensor:
                write_dict[loss_term] = test_loss[loss_term].item()
            else:
                write_dict[loss_term] = test_loss[loss_term]
    
    with open(save_dir + r'/config.yaml','w') as dumpfile:
        dumpfile.write(yaml.dump(write_dict))
    
    # save to csv
    if generate_csv:
        sum_dict = pd.DataFrame()
        sum_dict = sum_dict.append(write_dict, ignore_index=True)
        sum_dict.to_csv(os.path.join(save_dir, 'sum_test.csv'), index=False)
        sum_dict.to_csv(os.path.join(save_dir, f'{exp_name}_sum_test.csv'), index=False)

    # # st()
    # visualize test images
    if test_img != None:
        img_path_dict = {}
        for key in test_img.keys():
            img_path_dict[key] = os.path.join(save_dir, 'imgs', key)
            if not os.path.exists(img_path_dict[key]):
                os.makedirs(img_path_dict[key])
        # st()
        img_num = test_img['GT'].shape[0]

        hdr_list = ['GT', 'pred_hdr', 'GT_ibr_diff', 'GT_ibr_spec', 'pred_ibr_diff', 'pred_ibr_spec']
        sp_mask = generate_sphere_mask(test_img['GT'].shape[3])
        for idx in range(img_num):
            test_img['pred'][idx] = torch.mul(test_img['pred'][idx], sp_mask)
            test_img['pred_hdr'][idx] = torch.mul(test_img['pred_hdr'][idx], sp_mask)

        for key in test_img.keys():
            if key not in hdr_list:
                test_img[key] = map_hdr_clip(test_img[key])
            test_img[key] = test_img[key].cpu().detach().numpy()
            test_img[key] = np.transpose(test_img[key], (0, 2, 3, 1))
        # st()
        if not faltten_save:
            if img_num == len(name_ls):
            # assert img_num == len(name_ls)
                for idx in range(len(name_ls)):
                    for key in img_path_dict.keys():
                        if key in hdr_list and 'ibr' not in key:
                            write_image(os.path.join(img_path_dict[key], f'{name_ls[idx][0]}_{key}.hdr'), test_img[key][idx])
                        else:
                            write_image(os.path.join(img_path_dict[key], f'{name_ls[idx][0]}_{key}.png'), test_img[key][idx])
            else:
                for idx in range(img_num):
                    for key in img_path_dict.keys():
                        if key in hdr_list:
                            write_image(os.path.join(img_path_dict[key], f'{idx}_{key}.hdr'), test_img[key][idx])
                        else:
                            write_image(os.path.join(img_path_dict[key], f'{idx}_{key}.png'), test_img[key][idx])
        else:
            if img_num == len(name_ls):
            # assert img_num == len(name_ls)
                for idx in range(len(name_ls)):
                    for key in img_path_dict.keys():
                        if key in hdr_list:
                            write_image(
                                os.path.join(os.path.dirname(img_path_dict[key]), 
                                             f"{img_path_dict[key].split('/')[-1]}_{name_ls[idx][0]}_{key}.hdr"), 
                                test_img[key][idx])
                        else:
                            # write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{img_path_dict[key].split('/')[-1]}_{name_ls[idx][0]}_{key}.png"), test_img[key][idx])
                            write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{img_path_dict[key].split('/')[-1]}_{name_ls[idx][0]}_{key}.png"), srgb2linrgb(test_img[key][idx]))
                            
            else:
                print("img_num != len(name_ls), saving images with index")
                for idx in range(img_num):
                    for key in img_path_dict.keys():
                        # if key in hdr_list:
                        #     write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx}_{key}.hdr"), test_img[key][idx])
                        # else:
                        if key not in hdr_list:
                            if key == 'pred':
                                write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx:04d}_{key}.png"), test_img[key][idx])
                            elif key == 'face':
                                write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx:04d}_{key}.png"), linrgb2srgb(test_img[key][idx]))
                            else:
                                write_image(os.path.join(os.path.dirname(img_path_dict[key]), f"{idx:04d}_{key}.png"), srgb2linrgb(test_img[key][idx]))
    else:
        print("test_img is None, skipping saving test images")

def save_sep_test(save_dir, full_img, caption, name_loss_dict, options, num_test, exp_name=''):
    """
    save separate visualize and loss result for different envmaps.
    Args:
        save_dir: saving root dir;
        full_img: full visualize image, the order is in catption
        caption: the caption of image components in full_img;
        name_loss_dict: saved loss for each envmaps;
        options: test options
        num_test: total image number
    """
    # create saving directory
    img_hw = int(full_img.shape[1] / num_test)
    print("saving image size: ", img_hw)
    img_path_dict = {}
    for key in name_loss_dict.keys():
        if key != 'test_num':
            if not os.path.exists(os.path.join(save_dir, key)):
                os.makedirs(os.path.join(save_dir, key))
            img_path_dict[key] = os.path.join(save_dir, key)

    # st()
    if os.path.exists(os.path.join(save_dir,'sum_test.csv')):
        sum_dict = pd.read_csv(os.path.join(save_dir,'sum_test.csv'), index_col=0)
    else:
        sum_dict = pd.DataFrame()
    left_meter = 0
    for env_name in img_path_dict:
        print("saving {}".format(env_name))
        test_num = name_loss_dict[env_name]['test_num']
        name_loss_dict[env_name].pop('test_num')
        print("test_num: ", test_num)
        write_dict = {}
        write_dict['exp name'] = options.expname
        write_dict['test num'] = test_num
        write_dict['env name'] = env_name
        write_dict['k'] = name_loss_dict[env_name]['k']
        name_loss_dict[env_name].pop('k')
        # write_dict['caption'] = caption
        test_loss = name_loss_dict[env_name]
        for loss_term in test_loss:
            
            if type(test_loss[loss_term]) == torch.Tensor:
                write_dict[loss_term] = test_loss[loss_term].item() / test_num
            else:
                write_dict[loss_term] = test_loss[loss_term] / test_num

        sum_dict = sum_dict.append(write_dict, ignore_index=True)
        with open(os.path.join(img_path_dict[env_name], 'config.yaml'), 'w') as dumpfile:
            dumpfile.write(yaml.dump(write_dict))
    
        # write image
        env_img = full_img[:, int(left_meter): int(left_meter + img_hw * test_num), :]
        left_meter += img_hw * test_num
        
        # # add caption
        # st()
        # black_title = np.zeros((50, env_img.shape[2], 3))
        # cv2.putText(black_title, caption, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.65, (255, 255, 255), 2)
        # title = torch.tensor(black_title).permute(2, 0, 1).float()
        # env_img = torch.cat((title, env_img), dim=1)
        
        torchvision.utils.save_image(env_img, os.path.join(img_path_dict[env_name], f'{env_name}_test.png'))
    
    sum_dict.to_csv(os.path.join(save_dir, exp_name + 'sum_test.csv'), index=False)

def normalize(tensor):
    tensor = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False)(tensor)
    return tensor


if __name__ == '__main__':
    
    pred = torch.rand(7, 3, 64, 64)
    input = {'envmap': torch.rand(7, 3, 64, 64), 
                'shading': torch.rand(7, 3, 64, 64),
                'specular': torch.rand(7, 3, 64, 64),
                'normal': torch.rand(7, 3, 64, 64)}

    pred_render = {'specular': pred, 'shading': pred}
    gt_render = {'specular': pred, 'shading': pred}

    out = draw_outputs_full(pred, input, gt_render, pred_render)
