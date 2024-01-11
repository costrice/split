import os
import sys
from datetime import datetime
from typing import List

import torch
import torchvision.utils
import wandb
from dateutil import tz
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchsummary as summary
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

import torch
import torchvision.utils
import torchvision.transforms.functional as vtf
import wandb
from dateutil import tz
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.pix2pix as pix2pix
# from datasets.lightdataset import LightDataset
from lighting_estimation.codes.models.env_autoencoder import EnvAutoEncoder
from models.hrnet import get_hr_net
from utils.general import linrgb2srgb, srgb2linrgb, write_image, tensor2ndarray
from utils.warp import generate_sphere_mask_and_normal
from utils.ibrender import IBrender


def visualize(
        image_batch_list: List[torch.Tensor],
        visual_num: int = 16, 
        col_first: bool = False):
    group_size = 16

    # visualize in groups, each group arranges images horizontally, while
    # groups are arranged vertically
    for i in range(0, visual_num, group_size):
        i_upper = min(i + group_size, visual_num)
        group_grid = []
        black = None

        for image_batch in image_batch_list:
            image_grid = torchvision.utils.make_grid(
                image_batch[i:i_upper], nrow=group_size)
            black = torch.zeros_like(image_grid)
            group_grid.append(image_grid)

        group_grid = torch.cat(group_grid, dim=1)
        if i == 0:
            visual = group_grid
        else:
            visual = torch.cat((visual, black, group_grid), dim=1)

    visual = torch.clip(visual, 0, 1)
    return visual


def visualize_col(
        image_batch_list: List[torch.Tensor],
        visual_num: int = 16,):
    group_size = 4

    # visualize in groups, each group arranges images horizontally, while
    # groups are arranged vertically
    for i in range(0, visual_num, group_size):
        i_upper = min(i + group_size, visual_num)
        group_grid = []
        black = None
        # breakpoint()
        for image_batch in image_batch_list:
            image_grid = torchvision.utils.make_grid(image_batch[i:i_upper], nrow=1)
            black = torch.zeros_like(image_grid)
            group_grid.append(image_grid)
        
        group_grid = torch.cat(group_grid, dim=2)
        if i == 0:
            visual = group_grid
        else:
            visual = torch.cat((visual, black, group_grid), dim=2)

    visual = torch.clip(visual, 0, 1)
    return visual

def adjust_intensity(envmap_hdr, intensity_target=0.3, return_multiplier=False,
                     cmp_type="mean"):
    # compute spatial median or mean
    intensity = envmap_hdr.view(envmap_hdr.shape[0], 3, -1). \
        mean(dim=1, keepdim=True)
    if cmp_type == "median":
        intensity_cur = torch.median(intensity, dim=2, keepdim=True)
    elif cmp_type == "mean":
        intensity_cur = torch.mean(intensity, dim=2, keepdim=True)
    intensity_cur = intensity_cur.view(intensity_cur.shape[0], 1, 1, 1)
    multiplier = intensity_target / (intensity_cur + 1e-4)
    envmap_hdr = envmap_hdr * multiplier
    if return_multiplier:
        return envmap_hdr, multiplier
    else:
        return envmap_hdr


def tonemapping(envmap_hdr):
    # convert to mean = 0.3 and clip to [0, 1], then convert to sRGB
    envmap_ldr = adjust_intensity(envmap_hdr, return_multiplier=False)
    envmap_ldr = torch.clip(envmap_ldr, 0, 1)
    envmap_ldr = linrgb2srgb(envmap_ldr)
    return envmap_ldr

# def prepare_test_batch(gpu=0):
#     data_group = "indoor"
#     test_dataset = LightDataset(
#         root_dir=r'E:\Datasets',
#         split='test',
#         data_group=data_group,
#         input_compos='compo_sphere')
#     test_dataloader = DataLoader(
#         test_dataset,
#         shuffle=False,
#         num_workers=0,
#         batch_size=256,
#         pin_memory=True)
#     test_batch = next(iter(test_dataloader))
#     for compo in test_batch:
#         if isinstance(test_batch[compo], torch.Tensor):
#             test_batch[compo] = test_batch[compo].cuda(gpu, non_blocking=True)

#     if data_group == "outdoor":
#         # add sky
#         for compo in test_batch:
#             if isinstance(test_batch[compo], torch.Tensor):
#                 test_batch[compo] = test_batch[compo][:128]
#         test_dataset = LightDataset(
#             root_dir=r'E:\Datasets',
#             split='test',
#             data_group="sky",
#             input_compos='compo_sphere')
#         test_dataloader = DataLoader(
#             test_dataset,
#             shuffle=False,
#             num_workers=0,
#             batch_size=128,
#             pin_memory=True)
#         test_batch_sky = next(iter(test_dataloader))
#         for compo in test_batch_sky:
#             if isinstance(test_batch_sky[compo], torch.Tensor):
#                 test_batch[compo] = torch.cat(
#                     (test_batch[compo], test_batch_sky[compo].cuda(gpu)), dim=0)

#     return test_batch

    # with torch.no_grad():
    #     test_input_hdr = hdr_net_pred(test_batch) * sphere_mask
    #     test_real_hdr = test_batch['envmap'] * sphere_mask
    #
    # return test_input_hdr, test_real_hdr

def get_cloudy_mask(
    untex_hdr: torch.Tensor,
    ):
    b, c, h, w = untex_hdr.shape
    untex_hdr_adjusted, multiplier = adjust_intensity(
        untex_hdr, return_multiplier=True)

    # get LDR mask and compute difference on that mask
    # breakpoint()
    max_untex_light = untex_hdr_adjusted.mean(dim=1, keepdim=True)
    batch_max_value = max_untex_light.view(b, -1).max(dim=1, keepdim=True)[0]
    cloudy_thres = 10
    cloudy_mask = (batch_max_value.squeeze() < cloudy_thres)
    print(f"num_less: {sum(cloudy_mask)} in {b} batches")
    return cloudy_mask

def combine_ldr_and_hdr(
        untex_hdr: torch.Tensor,
        tex_ldr: torch.Tensor,
        mask_sp: torch.Tensor, 
        filter_cloudy: bool = False):
    b, c, h, w = untex_hdr.shape
    untex_hdr_adjusted, multiplier = adjust_intensity(
        untex_hdr, return_multiplier=True)

    # get LDR mask and compute difference on that mask
    # breakpoint()
    cloudy_mask = None
    if filter_cloudy:
        max_untex_light = untex_hdr_adjusted.mean(dim=1, keepdim=True)
        batch_max_value = max_untex_light.view(b, -1).max(dim=1, keepdim=True)[0]
        cloudy_thres = 20
        cloudy_mask = (batch_max_value.squeeze() < cloudy_thres)
        print(f"num_less: {sum(cloudy_mask)} in {b} batches")
        
    lowpart_mask = (untex_hdr_adjusted.mean(dim=1, keepdim=True) < 1).float()\
        .expand(b, c, h, w)
    lowpart_mask = vtf.gaussian_blur(lowpart_mask, kernel_size=3, sigma=1) * lowpart_mask
    lowpart_mask = lowpart_mask * mask_sp
    untex_lowpart = torch.clip(untex_hdr_adjusted, 0, 1) * lowpart_mask

    # # compute mean energy of ldr
    # total_energy = untex_hdr.view(b, c, -1).mean(dim=1, keepdim=True).sum(dim=2)
    # total_energy_ldr = (untex_hdr * untex_lowpart).view(b, c, -1)\
    #     .mean(dim=1,keepdim=True).sum(dim=2)
    # ldr_energy_ratio = total_energy_ldr / total_energy

    # adjust color of textured ldr according to the color of untextured ldr
    target_tex_mean_color = untex_lowpart.view(b, c, -1).sum(dim=2)
    lowpart_mask_sum = lowpart_mask.view(b, c, -1).sum(dim=2)
    target_tex_mean_color /= lowpart_mask_sum
    target_tex_mean_color = target_tex_mean_color[..., None, None]
    target_tex_mean_color_vis = target_tex_mean_color * lowpart_mask

    tex_lowpart = srgb2linrgb(tex_ldr) * lowpart_mask
    current_tex_mean_color = tex_lowpart.view(b, c, -1).sum(dim=2)
    current_tex_mean_color /= lowpart_mask_sum
    current_tex_mean_color = current_tex_mean_color[..., None, None]
    tex_lowpart_adjusted = tex_lowpart * target_tex_mean_color / \
              (current_tex_mean_color + 1e-6)
    # tex_lowpart_adjusted = tex_lowpart

    # texture_diff = tex_lowpart_adjusted - untex_lowpart
    # tex_hdr = (untex_hdr + texture_diff / multiplier).clip(min=0)
    tex_hdr = (1 - lowpart_mask) * untex_hdr + tex_lowpart_adjusted / multiplier
    tex_hdr_adjusted = tex_hdr * multiplier

    return tex_hdr, cloudy_mask

class SphericalLightingEstimator(nn.Module):
    def __init__(self, 
                 ckpt_est="",
                 ckpt_pred="", 
                 device='cpu', 
                 verbose=False, 
                 random_tex=False, 
                 filter_cloudy=False
                       ):
        super(SphericalLightingEstimator, self).__init__()

        # HDR Recovery Net for lighting estimation
        self.hdr_net = get_hr_net()

        # Texture Net for texture generation
        self.ldr_net = pix2pix.define_G(
            input_nc=3, output_nc=3, ngf=16, netG="local", gpu_ids=[0])


        hdr_net_ae = EnvAutoEncoder(vector_size=1024, sphere_size=64)
        self.hdr_net.decoder = hdr_net_ae.decoder
        # breakpoint()
        # self.ldr_net.load_state_dict(
        #     torch.load(ckpt_est, map_location='cpu')
        #     ["netG"])
        if os.path.exists(ckpt_est):
            self.load_state_dict(torch.load(ckpt_est))
            print(f"Loaded ckpt from {ckpt_est}.")
        # torch.save(self.state_dict(), ckpt_path)
        # breakpoint()
        if ckpt_pred:
            self.hdr_net.load_state_dict(
                torch.load(ckpt_pred,
                           map_location="cpu"), strict=False)
            print(f"ADDITIONAL Loaded pred ckpt from {ckpt_pred}.")
        self.hdr_net.decoder = hdr_net_ae.decoder
        self.mask_sp, _ = generate_sphere_mask_and_normal(64)
        self.mask_sp = torch.from_numpy(
            self.mask_sp.transpose(2, 0, 1)).float().to(device)
        # breakpoint()
        self.to(device)
        
        self.verbose = verbose
        self.random_tex = random_tex
        if self.random_tex:
            print("Using random texture for revison verbose only.")
            
        self.filter_cloudy = filter_cloudy
        if self.filter_cloudy:
            print("Using filter_cloudy for revison only, on outdoor evaluations")

    def forward(self, batch):
        
        untex_hdr = self.hdr_net(batch) * self.mask_sp
        untex_hdr = torch.exp(untex_hdr) - 1
        untex_ldr = tonemapping(untex_hdr)
        tex_ldr = self.ldr_net(untex_ldr)
        tex_ldr = torch.clip(tex_ldr, 0, 1) * self.mask_sp
        orig_tex_ldr = tex_ldr
        if self.random_tex:
            # shufffle tex_ldr along batch dimension
            # cya idea
            # tex_ldr = tex_ldr[torch.randperm(tex_ldr.size(0))]
            # ff idea
            gt = tonemapping(torch.clip(batch['envmap'], 0, 1) * self.mask_sp)
            tex_ldr = gt[torch.randperm(gt.size(0))]
            
        tex_hdr, cloudy_mask = combine_ldr_and_hdr(untex_hdr, tex_ldr, self.mask_sp, self.filter_cloudy)
        
        if self.verbose:
            orig_tex_hdr, _ = combine_ldr_and_hdr(untex_hdr, orig_tex_ldr, self.mask_sp, self.filter_cloudy)
            self.verbose_rand_tex('verbose_randtex', untex_hdr, tex_ldr, tex_hdr, orig_tex_ldr, orig_tex_hdr)
        if self.filter_cloudy:
            return untex_hdr, tex_ldr, tex_hdr, cloudy_mask
        return untex_hdr, tex_ldr, tex_hdr

    def verbose_rand_tex(self, 
                         save_path,
                         untex_hdr, 
                         tex_ldr, 
                         tex_hdr, 
                         orig_tex_ldr, 
                         orig_tex_hdr):
        os.makedirs(save_path, exist_ok=True)
        num_imgs = untex_hdr.shape[0]
        untex_hdr_adjusted = adjust_intensity(untex_hdr)
        # untex_ibr_adjusted = IBRRender.render_using_mirror_sphere_ext(
        # untex_hdr_adjusted, ibr_diff, ibr_spec)
        tex_hdr_adjusted = adjust_intensity(tex_hdr)
        orig_tex_hdr_adjusted = adjust_intensity(orig_tex_hdr)
        
        padding_size = 5
        black_padding = torch.zeros((num_imgs, 3, 64, padding_size)).to(untex_hdr.device)
        
        # write_image(os.path.join(save_path, 'rand_tex.png'),
        #             tensor2ndarray(visualize(
        #                 [linrgb2srgb(untex_hdr_adjusted.clip(0, 1)),
        #                  black_padding,
        #                 tex_ldr,
        #                 linrgb2srgb(tex_hdr_adjusted.clip(0, 1)),
        #                 orig_tex_ldr,
        #                 black_padding,
        #                 linrgb2srgb(orig_tex_hdr_adjusted.clip(0, 1)),
        #                 ], visual_num=num_imgs)))
        # breakpoint()
        save_tensor = tensor2ndarray(visualize_col([linrgb2srgb(untex_hdr_adjusted.clip(0, 1)),black_padding,tex_ldr,linrgb2srgb(tex_hdr_adjusted.clip(0, 1)),black_padding,orig_tex_ldr,linrgb2srgb(orig_tex_hdr_adjusted.clip(0, 1)),], visual_num=num_imgs))
        # save_tensor = np.transpose(save_tensor, (1, 0, 2))
        write_image(os.path.join(save_path, 'rand_tex_gt.png'), save_tensor)
        
        breakpoint()

if __name__ == '__main__':
    sle_net = SphericalLightingEstimator()
    sle_net.eval()
    test_batch = prepare_test_batch()
    IBRRender = IBrender(ridx=4, envmap_size=64,
                         bases_folder=r"E:\Codes\split_light\data_files")
    ibr_diff = IBRRender.transform_mat_diff
    ibr_spec = IBRRender.transform_mat_spec

    untex_hdr, tex_ldr, tex_hdr = sle_net(test_batch)

    untex_hdr_adjusted = adjust_intensity(untex_hdr)
    untex_ibr_adjusted = IBRRender.render_using_mirror_sphere_ext(
        untex_hdr_adjusted, ibr_diff, ibr_spec)
    tex_hdr_adjusted = adjust_intensity(tex_hdr)
    tex_ibr_adjusted = IBRRender.render_using_mirror_sphere_ext(
        tex_hdr_adjusted, ibr_diff, ibr_spec)

    envmap_hdr_adjusted = adjust_intensity(test_batch["envmap"])
    envmap_ibr_adjusted = IBRRender.render_using_mirror_sphere_ext(
        envmap_hdr_adjusted, ibr_diff, ibr_spec)

    write_image(r"E:\Codes\split_light\tmp\combining_indoor_linear_mean.png",
                tensor2ndarray(visualize(
                    [linrgb2srgb(untex_hdr_adjusted.clip(0, 1)),
                     linrgb2srgb(untex_ibr_adjusted["specular"].clip(0, 1)),
                     linrgb2srgb(untex_ibr_adjusted["diffuse"].clip(0, 1)),
                     # ldr_mask,
                     # linrgb2srgb(target_tex_mean_color_vis),
                     # tex_ldr,
                     # linrgb2srgb(tex_lowpart_adjusted),
                     linrgb2srgb(tex_hdr_adjusted.clip(0, 1)),
                     linrgb2srgb(tex_ibr_adjusted["specular"].clip(0, 1)),
                     linrgb2srgb(tex_ibr_adjusted["diffuse"].clip(0, 1)),
                     # linrgb2srgb(envmap_hdr_adjusted.clip(0, 1)),
                     # linrgb2srgb(envmap_ibr_adjusted["specular"].clip(0, 1)),
                     # linrgb2srgb(envmap_ibr_adjusted["diffuse"].clip(0, 1)),
                    ], visual_num=256)))



