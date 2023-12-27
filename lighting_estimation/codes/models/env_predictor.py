import os
from typing import OrderedDict
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

import math

from pdb import set_trace as st 




class FacePredictor(nn.Module):
    def __init__(self, train_setting=None, img_hw=256, code_scale=128):
        super().__init__()
        self.input_scale = img_hw
        self.code_scale = code_scale

        # self.multi_scale = multiscale
        # if self.multi_scale:
        #     print('Multi scale supervision. ')
        input_dim = 3 * len(train_setting)
        
        self.train_setting = train_setting
        for compo in train_setting:
            print(f'{compo} training')
        if 'mask' in train_setting:
            input_dim -= 2

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.2), 
            antialiased_cnns.BlurPool(32, stride=2)
        )
        # self.relu = nn.LeakyReLU(0.2)

        down_conv_num = int(math.log2(self.input_scale / 8))
        # print("down_conv_number")
        # print(down_conv_num)
        # self.down_conv_num = down_conv_num
        self.down_conv_ls = nn.ModuleList()
        # for i in range(down_conv_num):
        #     self.down_conv_ls.append(nn.Sequential(
        #         nn.Conv2d(16 * (2 ** i), 16 * (2 ** (i+1)), kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(16 * (2 ** (i+1))), 
        #         nn.LeakyReLU(0.2), 
        #         nn.Conv2d(16 * (2 ** (i+1)), 16 * (2 ** (i+1)), kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(16 * (2 ** (i+1))), 
        #         nn.LeakyReLU(0.2), 
        #         antialiased_cnns.BlurPool((16 * (2 ** (i+1))), stride=2)
        #     ))
        self.down_conv_ls.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2), 
            antialiased_cnns.BlurPool(64, stride=2)
        ))
        self.down_conv_ls.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 
            antialiased_cnns.BlurPool(128, stride=2)
        ))
        self.down_conv_ls.append(nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 
            antialiased_cnns.BlurPool(256, stride=2)
        ))
        # 2048

        # self.conv6 = nn.Conv2d(512, 512, kernel_size=8)
        # self.fc = nn.Linear((16 * 2 ** (down_conv_num + 2)), self.code_scale)
        self.bottleneck = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        
        # self.model = nn.Sequential(self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.relu, self.conv4, self.relu, self.conv5, self.relu, self.fc)
        
        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight, a=0.2)

        # self.encoder = LiNet_encoder(self.train_setting)
        # self.decoder = LiNet_decoder(multi_scale=self.multi_scale, output_scale=self.output_scale)
    
    def forward(self, x):
        input_ls = []
        for compo in self.train_setting:
            input_ls.append(x[compo])

        x = torch.cat(input_ls, dim=1)
        # st()
        x = self.conv1(x)
        for i in range(len(self.down_conv_ls)):
            x = self.down_conv_ls[i](x)

        # 64 x 4 x 4
        x = self.bottleneck(x)
        # st()
        return x

        
        
if __name__ == '__main__':
    # from .. import config
    training_need_ls_sp = ['shading', 'normal', 'specular', 'mask']
    sphere_scale = 64
    latent_scale = 1
    net_predictor = FacePredictor(train_setting=training_need_ls_sp, img_hw=sphere_scale, code_scale=latent_scale)
    
    input = {'shading': torch.randn(16, 3, 64, 64), 'normal': torch.randn(16, 3, 64, 64), 'specular': torch.randn(16, 3, 64, 64), 'mask': torch.randn(16, 1, 64, 64)}
    out = net_predictor(input)
    print(out.shape)

