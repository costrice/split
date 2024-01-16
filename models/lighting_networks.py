import torch
import torch.nn as nn
import torchvision.transforms.functional as vtf

import models.pix2pixHD as pix2pix
from models.env_autoencoder import EnvAutoEncoder
from models.hrnet import get_hr_net
from utils.general import linrgb2srgb, srgb2linrgb
from utils.geotransform import generate_sphere_mask_and_normal


def adjust_intensity(
    envmap_hdr, intensity_target=0.3, return_multiplier=False, cmp_type="mean"
):
    # compute spatial median or mean
    intensity = envmap_hdr.view(envmap_hdr.shape[0], 3, -1).mean(dim=1, keepdim=True)
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


def combine_ldr_and_hdr(
    untex_hdr: torch.Tensor, tex_ldr: torch.Tensor, mask_sp: torch.Tensor,
):
    b, c, h, w = untex_hdr.shape
    untex_hdr_adjusted, multiplier = adjust_intensity(untex_hdr, return_multiplier=True)

    lowpart_mask = (
        (untex_hdr_adjusted.mean(dim=1, keepdim=True) < 1).float().expand(b, c, h, w)
    )
    lowpart_mask = (
        vtf.gaussian_blur(lowpart_mask, kernel_size=3, sigma=1) * lowpart_mask
    )
    lowpart_mask = lowpart_mask * mask_sp
    untex_lowpart = torch.clip(untex_hdr_adjusted, 0, 1) * lowpart_mask

    # adjust color of textured ldr according to the color of untextured ldr
    target_tex_mean_color = untex_lowpart.view(b, c, -1).sum(dim=2)
    lowpart_mask_sum = lowpart_mask.view(b, c, -1).sum(dim=2)
    target_tex_mean_color /= lowpart_mask_sum
    target_tex_mean_color = target_tex_mean_color[..., None, None]
    # target_tex_mean_color_vis = target_tex_mean_color * lowpart_mask

    tex_lowpart = srgb2linrgb(tex_ldr) * lowpart_mask
    current_tex_mean_color = tex_lowpart.view(b, c, -1).sum(dim=2)
    current_tex_mean_color /= lowpart_mask_sum
    current_tex_mean_color = current_tex_mean_color[..., None, None]
    tex_lowpart_adjusted = (
        tex_lowpart * target_tex_mean_color / (current_tex_mean_color + 1e-6)
    )

    tex_hdr = (1 - lowpart_mask) * untex_hdr + tex_lowpart_adjusted / multiplier
    # tex_hdr_adjusted = tex_hdr * multiplier

    return tex_hdr


class LightingEstimator(nn.Module):
    def __init__(
        self, ckpt_path="", device="cuda:0",
    ):
        super(LightingEstimator, self).__init__()
        self.device = device

        # HDR light source net
        self.hdr_net = get_hr_net()

        # Ambient texture net
        self.ldr_net = pix2pix.define_G(
            input_nc=3, output_nc=3, ngf=16, netG="local", gpu_ids=[0]
        )

        hdr_net_ae = EnvAutoEncoder(vector_size=1024, sphere_size=64)
        self.hdr_net.decoder = hdr_net_ae.decoder

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt)
        print(f"Loaded ckpt from {ckpt_path}.")

        self.hdr_net.decoder = hdr_net_ae.decoder
        self.mask_sp, _ = generate_sphere_mask_and_normal(64)
        self.mask_sp = (
            torch.from_numpy(self.mask_sp.transpose(2, 0, 1)).float().to(self.device)
        )

        self.to(self.device)

    def forward(self, batch):
        untex_hdr = self.hdr_net(batch) * self.mask_sp
        untex_hdr = torch.exp(untex_hdr) - 1
        untex_ldr = tonemapping(untex_hdr)
        tex_ldr = self.ldr_net(untex_ldr)
        tex_ldr = torch.clip(tex_ldr, 0, 1) * self.mask_sp
        orig_tex_ldr = tex_ldr

        tex_hdr = combine_ldr_and_hdr(untex_hdr, tex_ldr, self.mask_sp)
        return untex_hdr, tex_ldr, tex_hdr
