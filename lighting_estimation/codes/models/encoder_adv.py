import os
from typing import OrderedDict
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

import math

from pdb import set_trace as st 
class LiNet_encoder(nn.Module):
    def __init__(self, train_setting):
        super().__init__()
        # self.albedo_train = albedo_train
        # self.normal_train = normal_train
        # self.licol_train = licol_train
        # self.shading_train = shading_train
        # self.specular_train = specular_train
        # self.face_train = face_train
        self.train_setting = train_setting

        # input_dim = 3 * (int(self.face_train) + int(self.albedo_train) + int(self.normal_train) + int(self.licol_train) + int(self.shading_train) + int(specular_train))

        input_dim = 3 * len(train_setting)
        if 'mask' in self.train_setting:
            input_dim -= 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), 
            antialiased_cnns.BlurPool(16, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            antialiased_cnns.BlurPool(32, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            antialiased_cnns.BlurPool(64, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            antialiased_cnns.BlurPool(128, stride=2)
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256), 
        #     antialiased_cnns.BlurPool(256, stride=2)
        # )
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=4)
        self.bottleneck = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # x = self.relu(self.conv5(x))
        # x = self.relu(self.conv6(x))
        # x = x[:, :, 0, 0]
        x = self.bottleneck(x)
        return x

class LiNet_decoder(nn.Module):
    def __init__(self, multi_scale=False, output_scale=32):
        super().__init__()

        up_scale_num = int(math.log2(output_scale / 8))
        self.upscale_num = up_scale_num
        self.multi_scale = multi_scale
        self.output_scale = output_scale # larger edge
        # Upscaling
        self.upconv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256) 
            )
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.upconv_dict = OrderedDict()
        cur_channel = 256
        cur_scale = 8
        for i in range(up_scale_num):
            upconv = nn.Sequential(
                nn.Conv2d(int(cur_channel), int(cur_channel / 2), kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(int(cur_channel / 2)),
            )

            up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.upconv_dict[f'{cur_scale}_upconv'] = upconv
            self.upconv_dict[f'{cur_scale}_up'] = up

            cur_channel /= 2
            cur_scale *= 2
        self.upconv_dict = nn.Sequential(self.upconv_dict)
        # st()
        self.deconv = nn.Conv2d(int(cur_channel), 4, kernel_size=3, stride=(1, 1), padding=1)

        if self.multi_scale:
            self.ms_ds4 = nn.Conv2d(int(cur_channel * 4), 3, kernel_size=3, stride=(1, 1), padding=1)
            self.ms_ds2 = nn.Conv2d(int(cur_channel * 2), 3, kernel_size=3, stride=(1, 1), padding=1)
            
        # st()
        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.xavier_uniform_(m[1].weight)
        
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # x = x[:, :, None, None]

        # x = torch.reshape(x, [x.shape[0], 64, 4, 4])
        x = self.relu(self.upconv(x))
        x = self.up1(x)
        # size of x: (batch, 256, 8, 8)
        x_ds2 = None
        x_ds4 = None

        for i in range(self.upscale_num):
            cur_scale = 8 * (2 ** i)
            if self.multi_scale:
                if cur_scale == self.output_scale / 4:
                    x_ds4 = self.ms_ds4(x)
                elif cur_scale == self.output_scale / 2:
                    x_ds2 = self.ms_ds2(x)

            # f'{cur_scale}_upconv'
            x = self.upconv_dict[i * 2](x)

            x = self.relu(x)
            # f'{cur_scale}_up'
            x = self.upconv_dict[i * 2 + 1](x)

        x = self.deconv(x)

        if self.multi_scale:
            return {'x': x, 'x_ds2': x_ds2, 'x_ds4': x_ds4}

        return {'x': x}

# class LiNet_decoder(nn.Module):
#     def __init__(self, multi_scale=False, output_scale=16):
#         super().__init__()

#         up_scale_num = int(math.log2(output_scale * 2 / 4))

#         # Upscaling
#         self.upconv = nn.Sequential(
#             nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), 
#             nn.BatchNorm2d(32) 
#             )
#         self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

#         self.upconv_ls = []
#         for i in range(up_scale_num):
#             upconv = nn.Sequential(
#                 nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), 
#                 nn.BatchNorm2d(8) ,
#                 nn.UpsamplingBilinear2d(scale_factor=2)
#             )
#             self.upconv_ls.append()
#         self.deconv2 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), 
#             nn.BatchNorm2d(16) 
#             )
#         self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
#         self.deconv3 = nn.Sequential(
#             nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), 
#             nn.BatchNorm2d(8) 
#             )
#         self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

#         self.deconv4 = nn.Conv2d(8, 4, kernel_size=3, stride=(2, 1), padding=1)

#         self.multi_scale = multi_scale
#         self.output_scale = output_scale
#         if self.multi_scale:
#             self.ms_b1 = nn.Conv2d(32, 3, kernel_size=3, stride=(2, 1), padding=1)
#             self.ms_b2 = nn.Conv2d(16, 3, kernel_size=3, stride=(2, 1), padding=1)
            
#         # st()
#         for m in enumerate(self.modules()):
#             if isinstance(m[1], nn.Conv2d):
#                 nn.init.xavier_uniform_(m[1].weight)
        
#         self.relu = nn.LeakyReLU(0.2)
        
#     def forward(self, x):
#         x = x[:, :, None, None]

#         x = torch.reshape(x, [x.shape[0], 64, 4, 4])
#         x = self.relu(self.deconv1(x))
#         x = self.up1(x)
#         # size of x: (batch, 32, 8, 8)
#         if self.multi_scale:
#             x_b1 = self.ms_b1(x)
        
#         x = self.relu(self.deconv2(x))
#         x = self.up2(x)
#         # size of x: (batch, 16, 16, 16)
#         if self.multi_scale:
#             x_b2 = self.ms_b2(x)
        
#         x = self.relu(self.deconv3(x))
#         x = self.up3(x)
#         # size of x: (batch, 8, 32, 32)

#         x = self.deconv4(x)

#         if self.multi_scale:
#             return {'x': x, 'x_ds2': x_b1, 'x_ds4': x_b2}

#         return {'x': x}


class LiNet(nn.Module):
    def __init__(self, multiscale=False, train_setting=None, img_hw=256, envmap_scale=16):
    # def __init__(self, multiscale=False, albedo_train=False, normal_train=False, licol_train=False, shading_train=False, specular_train=False, face_train=True, img_hw=256, envmap_scale=16):
        super().__init__()
        self.input_scale = img_hw
        self.output_scale = envmap_scale

        self.multi_scale = multiscale
        if self.multi_scale:
            print('Multi scale supervision. ')

        self.train_setting = train_setting
        # if 'mask' not in self.train_setting:
        #     self.train_setting.append('mask')
            
        for compo in train_setting:
            print(f'{compo} training')

        
        # self.albedo_train = albedo_train
        # if self.albedo_train:
        #     print('Albedo training.')

        # self.normal_train = normal_train
        # if self.normal_train:
        #     print('Normal training.')

        # self.licol_train = licol_train
        # if self.licol_train:
        #     print('licol training.')

        # self.shading_train = shading_train
        # if self.shading_train:
        #     print('shading training.')

        # self.specular_train = specular_train
        # if self.specular_train:
        #     print('specular training.')
            
        # self.face_train = face_train
        # if self.face_train:
        #     print('face training.')

        self.encoder = LiNet_encoder(self.train_setting)
        self.decoder = LiNet_decoder(multi_scale=self.multi_scale, output_scale=self.output_scale)
    
    def forward(self, x):
        input_ls = []
        # st()
        for compo in self.train_setting:
            input_ls.append(x[compo])
        # if self.face_train:
        #     input_ls.append(x['face'])
        # if self.albedo_train:
        #     input_ls.append(x['albedo'])
        # if self.normal_train:
        #     input_ls.append(x['normal'])
        # if self.licol_train:
        #     input_ls.append(x['light_color'])
        # if self.shading_train:
        #     input_ls.append(x['shading'])
        # if self.specular_train:
        #     input_ls.append(x['specular'])

        x = torch.cat(input_ls, dim=1)

        latent_vec = self.encoder(x)
        pred = self.decoder(latent_vec)
        x_hat = pred['x']
        if self.multi_scale:
            x_hat_ds2, x_hat_ds4 = pred['x_ds2'], pred['x_ds4']


        confidence = F.softmax(x_hat[:, :1], dim=1)
        log_lighting = torch.nn.Softplus()(x_hat[:, 1:])  # = log(1 + est_lighting) > 0
        # log_lighting = torch.sum(confidence * log_lighting, dim=4)
        log_lighting = confidence * log_lighting

        return log_lighting, x_hat_ds2, x_hat_ds4
        # if self.multi_scale:
        #     return {'fs': log_lighting, 'ds2': x_hat_ds2, 'ds4': x_hat_ds4}
        # return {'fs': log_lighting}

class MS_Discriminator(nn.Module):
    def __init__(self, output_scale):
        super().__init__()

        self.output_scale = output_scale
        self.from_rgb1 = nn.Sequential(
            nn.Conv2d(3, 32, (1, 1), bias=True), 
            nn.BatchNorm2d(32))
        self.from_rgb2 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1), bias=True), 
            nn.BatchNorm2d(64))
        self.from_rgb3 = nn.Sequential(
            nn.Conv2d(3, 128, (1, 1), bias=True), 
            nn.BatchNorm2d(128))

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
        )

        # self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(8, 16))
        # self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(4, 8))
        # self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(int(output_scale / 4), int(output_scale / 2)))
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(int(output_scale / 8), int(output_scale / 4)))
        # self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))


        self.fc = nn.Linear(int(256 * self.output_scale * self.output_scale / 4 / 8), 1)
        self.elu = nn.ELU()

    def forward(self, input_dict):
        """
        forward pass of the discriminator
        :param inputs: dict of multiscale input images
        :return raw prediction values
        """
        
        x = input_dict['pred']
        x_ds2 = input_dict['pred_ds2']
        x_ds4 = input_dict['pred_ds4']

        fea_16 = self.elu(self.from_rgb1(x))
        fea_8 = self.elu(self.from_rgb2(x_ds2))
        fea_4 = self.elu(self.from_rgb3(x_ds4))

        out = self.elu(self.layer1(fea_16))
        out = self.avgpool1(out)
        out = torch.cat([out, fea_8], dim=1)
        
        out = self.elu(self.layer2(out))
        out = self.avgpool2(out)
        out = torch.cat([out, fea_4], dim=1)

        out = self.elu(self.layer3(out))
        # out = self.maxpool3(out)
        # st()
        out = out.view(out.size(0), -1)
        # st()
        out = self.fc(out)
        out = out.view(-1)
        
        return out
        


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.from_rgb1 = nn.Sequential(
            nn.Conv2d(3, 32, (1, 1), bias=True), 
            nn.BatchNorm2d(32))
        # self.from_rgb2 = nn.Sequential(
        #     nn.Conv2d(3, 64, (1, 1), bias=True), 
        #     nn.BatchNorm2d(64))
        # self.from_rgb3 = nn.Sequential(
        #     nn.Conv2d(3, 128, (1, 1), bias=True), 
        #     nn.BatchNorm2d(128))

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
        )

        self.maxpool1 = nn.AdaptiveMaxPool2d(output_size=(8, 16))
        self.maxpool2 = nn.AdaptiveMaxPool2d(output_size=(4, 8))
        self.maxpool3 = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc = nn.Conv2d(64, 1, (1, 1), bias=True)
        self.elu = nn.ELU()

    def forward(self, input_dict):
        """
        forward pass of the discriminator
        :param inputs: dict of multiscale input images
        :return raw prediction values
        """
        if isinstance(input_dict, dict):
            input_dict = input_dict['pred']
            
        x_16 = input_dict

        fea_16 = self.elu(self.from_rgb1(x_16))

        out = self.elu(self.layer1(fea_16))
        out = self.maxpool1(out)
        # out = torch.cat([out, fea_8], dim=1)
        
        out = self.elu(self.layer2(out))
        out = self.maxpool2(out)
        # out = torch.cat([out, fea_4], dim=1)

        out = self.elu(self.layer3(out))
        out = self.maxpool3(out)

        out = self.fc(out)
        out = out.view(-1)
        
        return out
        
        

        
