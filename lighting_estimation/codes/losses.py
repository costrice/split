from os import device_encoding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T
from pdb import set_trace as st
from utils.utils_lighting import generate_real_dict

import torchvision
import torch.utils.model_zoo as model_zoo
from utils.utils_lighting import map_hdr, generate_sphere_mask

get_into_flag = 0

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        my_vgg = torchvision.models.vgg16(pretrained=False)
        pre = torch.load('/data4/chengyean/data/vgg16-397923af.pth')
        my_vgg.load_state_dict(pre)
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(my_vgg.features[:4].eval())
        blocks.append(my_vgg.features[4:9].eval())
        blocks.append(my_vgg.features[9:16].eval())
        blocks.append(my_vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y) 
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

def mean_Lp_loss_masked(p):
    """
    A closure that returns a function computing the mean of power to the p of differences between masked label and
    prediction.
    Args:
        p: the exponent of power
    Returns:
        the mean of power to the p of differences between masked label and prediction.
    """
    
    def functional(est, gt, mask):
        """
        Functional part in the closure.
        """
        return torch.sum(torch.pow(torch.abs((gt - est) * mask), exponent=p)) / torch.sum(mask)
    
    return functional

def log_l1_distance():
    # weight = torch.zeros(1, int(scale / 2), scale).to(device)
    # for lat in range(int(scale / 2)):
    #     weight[:, lat, :] = np.cos((lat + 0.5 - int(scale / 4)) / int(scale / 4) * np.pi / 2)

    def functional(log_est, gt):
        try:
            assert torch.min(gt) >= 0
        except AssertionError:
            print(torch.min(gt))
            print(gt)
        # breakpoint()
        loss = torch.mean(torch.abs((torch.log(1. + gt) - log_est)))
        return loss

    return functional


def log_perceputual(device):
    # weight = torch.zeros(1, int(scale / 2), scale).to(device)
    # for lat in range(int(scale / 2)):
    #     weight[:, lat, :] = np.cos((lat + 0.5 - int(scale / 4)) / int(scale / 4) * np.pi / 2)
    vgg_loss = VGGPerceptualLoss().to(device)
    def functional(log_est, gt):
        try:
            assert torch.min(gt) >= 0
        except AssertionError:
            print(torch.min(gt))
            print(gt)
            st()
        # st()
        loss = vgg_loss((torch.exp(log_est) - 1), gt)
        return loss

    return functional


# def loss_linet(device):
#     loss_log_l1 = log_l1_distance()
#     # loss_log_perceptual = weighted_log_perceputual(scale, device)
#     # loss_log_perceptual = log_perceputual(device)

#     def functional(net_output, input_batch):
#         real_samps = generate_real_dict(input_batch)

#         loss_l1_fs = loss_log_l1(net_output['pred'], real_samps['pred'])
#         loss_l1_ds2 = loss_log_l1(net_output['ds2'], real_samps['ds2'])
#         loss_l1_ds4 = loss_log_l1(net_output['pred_ds4'], real_samps['pred_ds4'])
#         # loss_vgg = loss_log_perceptual(net_output['pred'], real_samps['pred'])
        
#         loss_l1 = 0.6 * loss_l1_fs + 0.2 * loss_l1_ds2 + 0.199 * loss_l1_ds4
#         loss_total = loss_l1
#         return {
#             "total": loss_total,
#             "l1_fs": loss_l1_fs, 
#             "l1_ds2": loss_l1_ds2, 
#             "l1_ds4": loss_l1_ds4, 
#             "l1": loss_l1,
#             # "vgg": loss_vgg
#         }
#     return functional
def google_to_ldr(hdr):
    # soft clipping
    # st()
    hdr = 1 - 1 / 40 * torch.log(1. + torch.exp(-40 * (hdr - 1)))
    hdr = torch.clip(hdr, 1e-8, 1)
    # gamma correction
    # st()
    hdr = hdr.pow(1 / 2.2)
    return hdr


def loss_google(ibr, device):
    loss_l1_fun = nn.L1Loss()
    mask_sp = generate_sphere_mask(out_hw=32, device=device)
    mask_sp_ds2 = generate_sphere_mask(out_hw=16, device=device)
    mask_sp_ds4 = generate_sphere_mask(out_hw=8, device=device)
    mask_sp_ds8 = generate_sphere_mask(out_hw=4, device=device)
    mask_dict = {'fs': mask_sp, 'ds2': mask_sp_ds2, 'ds4': mask_sp_ds4, 'ds8': mask_sp_ds8}
    def functional(pred_render_dict, gt_render_dict):
        # st()
        loss_l1 = 0.0
        for key1 in mask_dict.keys():
            for key2 in pred_render_dict[key1].keys():
                # st()
                pred_render_dict[key1][key2] = torch.mul(pred_render_dict[key1][key2], mask_dict[key1])
                pred_render_dict[key1][key2] = google_to_ldr(pred_render_dict[key1][key2])
                pred_render_dict[key1][key2] = torch.clip(pred_render_dict[key1][key2], 1e-8, 1)

                gt_render_dict[key1][key2] = torch.mul(gt_render_dict[key1][key2], mask_dict[key1])
                gt_render_dict[key1][key2] = google_to_ldr(gt_render_dict[key1][key2])
                gt_render_dict[key1][key2] = torch.clip(gt_render_dict[key1][key2], 1e-8, 1)

            loss_scale = 0.2 * loss_l1_fun(pred_render_dict[key1]['mirror'], gt_render_dict[key1]['mirror'])
            loss_scale += 0.6 * loss_l1_fun(pred_render_dict[key1]['diffuse'], gt_render_dict[key1]['diffuse'])
            loss_scale += 0.2 * loss_l1_fun(pred_render_dict[key1]['specular'], gt_render_dict[key1]['specular'])
            loss_l1 += loss_scale
        # st()

        return {
            "l1": loss_l1, 
            "total": loss_l1
        }
    return functional


def loss_linet(ibr, device):
    loss_log_l1 = log_l1_distance()

    # loss_log_perceptual = weighted_log_perceputual(scale, device)
    # loss_log_perceptual = log_perceputual(device)
    loss_fun_l1 = nn.L1Loss()
    if ibr:
        ratio = [0.2, 0.2, 0.6]
    else:
        ratio = [1, 0, 0]

    mask_sp = generate_sphere_mask(out_hw=64, device=device)
    mask_sp_ds2 = generate_sphere_mask(out_hw=32, device=device)
    mask_sp_ds4 = generate_sphere_mask(out_hw=16, device=device)
    mask_sp_ds8 = generate_sphere_mask(out_hw=8, device=device)
    def functional(net_output, real_samps):

        # real_samps = generate_real_dict(gt)
        # st()
        net_output['fs'] = torch.mul(net_output['fs'], mask_sp)
        net_output['ds2'] = torch.mul(net_output['ds2'], mask_sp_ds2)
        net_output['ds4'] = torch.mul(net_output['ds4'], mask_sp_ds4)
        real_samps['fs'] = torch.mul(real_samps['fs'], mask_sp)
        real_samps['ds2'] = torch.mul(real_samps['ds2'], mask_sp_ds2)
        real_samps['ds4'] = torch.mul(real_samps['ds4'], mask_sp_ds4)
        pred_ibr["diffuse"] = torch.mul(pred_ibr["diffuse"], mask_sp)
        pred_ibr["specular"] = torch.mul(pred_ibr["specular"], mask_sp)
        gt_ibr["diffuse"] = torch.mul(gt_ibr["diffuse"], mask_sp)
        gt_ibr["specular"] = torch.mul(gt_ibr["specular"], mask_sp)

        loss_l1_fs = loss_log_l1(net_output['fs'], real_samps['fs'])
        loss_l1_ds2 = loss_log_l1(net_output['ds2'], real_samps['ds2'])
        loss_l1_ds4 = loss_log_l1(net_output['ds4'], real_samps['ds4'])
        # loss_vgg = loss_log_perceptual(net_output['pred'], real_samps['pred'])
        
        loss_l1 = 0.6 * loss_l1_fs + 0.2 * loss_l1_ds2 + 0.199 * loss_l1_ds4

        loss_l1_ibr_diff = loss_fun_l1(pred_ibr["diffuse"], gt_ibr["diffuse"])
        loss_l1_ibr_spec = loss_fun_l1(pred_ibr["specular"], gt_ibr["specular"])
        loss_l1_ibr = ratio[1]  * loss_l1_ibr_diff + ratio[2] * loss_l1_ibr_spec

        loss_total = ratio[0] * loss_l1 + loss_l1_ibr
        return {
            "total": loss_total,
            "l1_fs": loss_l1_fs, 
            "l1_ds2": loss_l1_ds2, 
            "l1_ds4": loss_l1_ds4, 
            "l1": loss_l1,
            "ibr_diff": loss_l1_ibr_diff,
            "ibr_spec": loss_l1_ibr_spec,
            "ibr": loss_l1_ibr,
            # "vgg": loss_vgg
        }
    return functional

def loss_autoencoder(ibr, device, ldr_train=0, loss_ratio=None):
    # TODO change scale
    loss_recon = log_l1_distance()
    # loss_recon = nn.L1Loss()
    # loss_perceptual = log_perceputual(device=device)
    loss_fun_l1 = nn.L1Loss()
    if ibr:
        if loss_ratio == None:
            loss_ratio = [0.6, 0.2, 0.2]
        ratio = loss_ratio
    else:
        ratio = [1, 0, 0]
    mask_sp = generate_sphere_mask(out_hw=64, device=device)
    # st()
    def functional(pred, gt, pred_ibr, gt_ibr, epoch=10):
        # TODO outdoor envmap sun may require different loss function
        if ldr_train:
            pred = torch.clip(pred, 0, 1)
            gt = torch.clip(gt, 0, 1)
            pred_ibr["diffuse"] = torch.clip(pred_ibr["diffuse"], 0, 1)
            pred_ibr["specular"] = torch.clip(pred_ibr["specular"], 0, 1)
            gt_ibr["diffuse"] = torch.clip(gt_ibr["diffuse"], 0, 1)
            gt_ibr["specular"] = torch.clip(gt_ibr["specular"], 0, 1)

        pred = torch.mul(pred, mask_sp)
        gt = torch.mul(gt, mask_sp)
        pred_ibr["diffuse"] = torch.mul(pred_ibr["diffuse"], mask_sp)
        pred_ibr["specular"] = torch.mul(pred_ibr["specular"], mask_sp)
        gt_ibr["diffuse"] = torch.mul(gt_ibr["diffuse"], mask_sp)
        gt_ibr["specular"] = torch.mul(gt_ibr["specular"], mask_sp)
            
        loss_l1 = loss_recon(pred, gt)
        # loss_vgg = loss_perceptual(net_output, input_batch)
        loss_l1_ibr_diff = loss_fun_l1(pred_ibr["diffuse"], gt_ibr["diffuse"])
        loss_l1_ibr_spec = loss_fun_l1(pred_ibr["specular"], gt_ibr["specular"])
        loss_l1_ibr = ratio[1] * loss_l1_ibr_diff + ratio[2] * loss_l1_ibr_spec

        # ibr warmup
        loss_total = ratio[0] * loss_l1 + loss_l1_ibr

        # if epoch > 5:
        #     loss_total = ratio[0] * loss_l1 + ratio[1] * loss_l1_ibr
        # else:
        #     loss_total = loss_l1

        return {
            "total": loss_total,
            "recon": 1 * loss_l1, 
            "ibr_diff": loss_l1_ibr_diff,
            "ibr_spec": loss_l1_ibr_spec,
            "ibr": loss_l1_ibr,
            # "vgg": 0.0 * loss_vgg
        }
    
    return functional

def loss_predictor(ibr, device):
    loss_recon_code = nn.L1Loss()
    loss_recon_env = log_l1_distance()
    # loss_perceputual_env = log_perceputual(device=device)
    mask_sp = generate_sphere_mask(out_hw=64, device=device)
    def functional(pred_code, pred_envmap, gt_envmap, rec_code):
        loss_l1_code = loss_recon_code(pred_code, rec_code)

        
        pred_envmap = torch.mul(pred_envmap, mask_sp)
        gt_envmap = torch.mul(gt_envmap, mask_sp)
        loss_l1_env = loss_recon_env(pred_envmap, gt_envmap)
        # loss_vgg_env = loss_perceputual_env(pred_envmap, batch)
        loss_total = loss_l1_code + 10 ** (-3) * (1 * loss_l1_env)
        # loss_total = loss_l1_code + 10 ** (-3) * (1 * loss_l1_env + 0.0 * loss_vgg_env)
        return {
            "total": loss_total,
            "l1_code": loss_l1_code,
            "l1_env": 1 * loss_l1_env, 
            # "vgg_env": 0 * loss_vgg_env
        }
    
    return functional

def high2code(gt_high, device):
    l_high = torch.zeros(gt_high.shape[0], 5).to(device)
    # st()
    h = gt_high.shape[2]
    w = gt_high.shape[3]
    for b in range(gt_high.shape[0]):
        # find the max r,g,b value using Levenberg–Marquardt
        max_i, max_j = 0, 0
        for c in range(3):
            temp = gt_high[b, c, :, :]
            index = torch.argmax(temp)
            if (index + 1) % w == 0:
                i = (index + 1) // w - 1
                j = w - 1
            else:
                i = (index + 1) // w
                j = (index + 1) % w - 1
            max_i += i
            max_j += j
            l_high[b, c + 2] = temp[i, j]
        # max_i /= 3
        # max_j /= 3

        # convert to [0, 1]
        l_high[b, 0] = max_i / h / 3
        l_high[b, 1] = max_j / w / 3

        # # convert to log-space
        # l_high[b, 2] = torch.log(l_high[b, 2] + 1e-6)
        # l_high[b, 3] = torch.log(l_high[b, 3] + 1e-6)
        # l_high[b, 4] = torch.log(l_high[b, 4] + 1e-6)

        # st()
    return l_high

def loss_tdv(device):
    loss_l1_fun = nn.L1Loss()
    loss_mse_fun = nn.MSELoss()
    mask_sp = generate_sphere_mask(out_hw=64, device=device)
    
    weight = torch.zeros(1, 256, 512).to(device)
    for lat in range(256):
        weight[:, lat, :] = np.cos((lat + 0.5 - 128) / 128 * np.pi / 2)
    def functional(z_low_ae, z_low_pred, z_high_pred, rec_envmap_ldr, pred_envmap, gt):
        loss_total = 0

        gt_ldr = torch.clip(gt, 0, 1)
        gt_high = gt - gt_ldr
        z_high_gt = high2code(gt_high, device=device)
        # code loss:
        loss_z_low = loss_mse_fun(z_low_pred, z_low_ae)
        loss_total += loss_z_low

        z_high_pred = torch.squeeze(z_high_pred)
        loss_z_high = loss_mse_fun(z_high_pred, z_high_gt)
        loss_total += 0.0001 * loss_z_high
        
        # envmap sp loss:
        # rec_pred_ldr = torch.mul(rec_envmap_ldr, mask_sp)
        # gt_ldr = torch.mul(gt, mask_sp)

        # envmap rect loss:
        rec_pred_ldr = torch.mul(rec_envmap_ldr, weight)
        gt_ldr = torch.mul(gt_ldr, weight)

        loss_env_rec_ldr = loss_l1_fun(rec_pred_ldr, gt_ldr)
        loss_total += loss_env_rec_ldr

        # envmap sp loss:
        # pred = torch.mul(pred_envmap, mask_sp)
        # gt = torch.mul(gt, mask_sp)

        # envmap rect loss:
        pred = torch.mul(pred_envmap, weight)
        gt = torch.mul(gt, weight)

        loss_env_rec = loss_l1_fun(pred, gt)
        loss_total += loss_env_rec
        return {
            "total": loss_total,
            "z_low": loss_z_low,
            "z_high": loss_z_high,
            "env_rec_ldr": loss_env_rec_ldr,
            "env_rec": loss_env_rec
        }
    return functional


class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(dis)
        self.device = device

    def dis_loss(self, real_samps, fake_samps):

        # device for computations:
        device = self.device

        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(torch.nn.ReLU()(1 + r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 - f_r_diff)))

class StandardGANLoss(GANLoss):
    def __init__(self, deivce, dis):
        
        super().__init__(dis)

        self.criterion = nn.BCEWithLogitsLoss()
        self.device = deivce

    def dis_loss(self, real_samps, fake_samps):


        # device for computations:
        device = self.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps['pred'].shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps['pred'].shape[0]).to(device))

        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        preds = self.dis(fake_samps)
        return self.criterion(torch.squeeze(preds), torch.ones(fake_samps['pred'].shape[0]).to(self.device))


class WGAN_GP(GANLoss):

    def __init__(self, deivce, dis, drift=0.001, use_gp=True):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp
        self.device = deivce

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        # device for computations:
        device = self.device

        batch_size = real_samps['pred'].shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

        # create the merge of both real and fake samples

        merged = (epsilon * real_samps['pred']) + ((1 - epsilon) * fake_samps['pred'])
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,  grad_outputs=torch.ones_like(op), create_graph=True, retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps)
        real_out = self.dis(real_samps)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps))

        return loss

if __name__ == '__main__':
    # test functionality
    loss = loss_autoencoder(ibr=True, device='cpu', ldr_train=True)
    pred = torch.ones(16, 3, 64, 64)
    gt = torch.zeros(16, 3, 64, 64)
    pred_ibr = {"diffuse": torch.randn(16, 3, 64, 64), "specular": torch.randn(16, 3, 64, 64)}
    gt_ibr = {"diffuse": torch.randn(16, 3, 64, 64), "specular": torch.randn(16, 3, 64, 64)}
    print(loss(pred, gt, pred_ibr, gt_ibr))
