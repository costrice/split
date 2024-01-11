import os
from typing import OrderedDict
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns

import math

from pdb import set_trace as st 
class Envmap_encoder(nn.Module):
    def __init__(self, vector_size=64, sphere_size=128):
        super().__init__()
        self.out_vec = vector_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2), 
            nn.BatchNorm2d(16), 
            antialiased_cnns.BlurPool(16, stride=2)
        )
        # print(in_size)
        down_conv_num = int(math.log2(sphere_size / 8))
        #down_conv_num: 3
        print("down_conv_number")
        print(down_conv_num)
        self.down_conv_ls = nn.ModuleList()

        for i in range(down_conv_num):
            self.down_conv_ls.append(nn.Sequential(
                nn.Conv2d(16 * (2 ** i), 16 * (2 ** (i+1)), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16 * (2 ** (i+1))), 
                nn.LeakyReLU(0.2), 
                nn.Conv2d(16 * (2 ** (i+1)), 16 * (2 ** (i+1)), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16 * (2 ** (i+1))), 
                nn.LeakyReLU(0.2), 
                antialiased_cnns.BlurPool((16 * (2 ** (i+1))), stride=2)
            ))

        # 0127 2048 train
        # self.bottleneck = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        
        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, x):
        x = self.conv1(x)
        for i in range(len(self.down_conv_ls)):
            x = self.down_conv_ls[i](x)
        # x = x.view(x.shape[0], -1)
        x = self.bottleneck(x)
        return x

class Envmap_decoder(nn.Module):
    def __init__(self, vector_size=64, sphere_size=128, use_sphere=False):
        super().__init__()
        self.input_size = vector_size
        self.sphere_size = sphere_size

        self.upconv1 = nn.Sequential(
            nn.Conv2d(int(vector_size / 16), 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=2)
            )

        self.upconv_list = nn.ModuleList()
        # TODO Change this to a list of upsampling layers
        up_num = int(math.log(sphere_size / 8, 2))
        for i in range(up_num):
            self.upconv_list.append(
                nn.Sequential(
                nn.Conv2d(int(256 / (2**i)), int(256 / (2**(i + 1))), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(256 / (2**(i + 1)))),
                nn.LeakyReLU(0.2),
                nn.Conv2d(int(256 / (2**(i + 1))), int(256 / (2**(i + 1))), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(int(256 / (2**(i + 1)))),
                nn.LeakyReLU(0.2),
                nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )
        
        self.out_layer = nn.Sequential(
            nn.Conv2d(int(256 / (2**up_num)), 3, kernel_size=3, stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(3),
        )

        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')

    
    def forward(self, x):
        # st()
        # x = x.reshape(x.shape[0], int(x.shape[1] / 16), 4, 4)
        x = self.upconv1(x)
        for i in range(len(self.upconv_list)):
            x = self.upconv_list[i](x)

        x = self.out_layer(x)
        return x


class Envmap_decoder_ms(nn.Module):
    def __init__(self, vector_size=64, sphere_size=128, use_sphere=False):
        super().__init__()
        self.input_size = vector_size
        self.sphere_size = sphere_size

        self.upconv1 = nn.Sequential(
            nn.Conv2d(int(vector_size / 16), 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2), 
            nn.UpsamplingBilinear2d(scale_factor=2)
            )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2), 
            nn.UpsamplingBilinear2d(scale_factor=2)
            )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
            nn.UpsamplingBilinear2d(scale_factor=2)
            )
        self.upconv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), 
            nn.UpsamplingBilinear2d(scale_factor=2)
            )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.out_layer_ds2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=(1, 1), padding=0),
            nn.ReLU()
        )
        self.out_layer_ds4 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, stride=(1, 1), padding=0),
            nn.ReLU()
        )

        for m in enumerate(self.modules()):
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')

    
    def forward(self, x):
        # x = x.reshape(x.shape[0], int(x.shape[1] / 16), 4, 4)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x_ds4 = self.out_layer_ds4(x)
        x = self.upconv3(x)
        x_ds2 = self.out_layer_ds2(x)
        x = self.upconv4(x)
        x = self.out_layer(x)

        return x, x_ds2, x_ds4




class EnvAutoEncoder(nn.Module):
    def __init__(self, vector_size=128, sphere_size=64):
        super().__init__()
        self.vector_size = vector_size
        self.sphere_size = sphere_size

        print("vector_size")
        print(self.vector_size)
        print("envmap_size")
        print(self.sphere_size)
        
        self.encoder = Envmap_encoder(vector_size=self.vector_size, sphere_size=self.sphere_size)
        self.decoder = Envmap_decoder_ms(vector_size=self.vector_size, sphere_size=self.sphere_size)
        # self.decoder = Envmap_decoder(vector_size=self.vector_size, sphere_size=self.sphere_size)

    def forward(self, x):
        z = self.encoder(x)
        # st()
        out = self.decoder(z)
        return out

    def encoder_forward(self, x):
        z = self.encoder(x)
        return z

    def decoder_forward(self, z):
        out = self.decoder(z)
        if type(out) == tuple:
            out = out[0]
        return out



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.lrelu = nn.LeakyReLU(0.2)

        self.from_rgb1 = nn.Sequential(
            nn.Conv2d(3, 16, (1, 1), bias=True))
        # self.from_rgb2 = nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=True))
        # self.from_rgb3 = nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=True))

        self.layer0 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(16, 32, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32, 64, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, (8, 8), padding=0, bias=True),
            nn.LeakyReLU(0.2)
            )
        self.fc = nn.Linear(128, 1, bias=True)


    def forward(self, x_fs):
        """
        forward pass of the discriminator
        :param inputs: dict of multiscale input images
        :return raw prediction values
        """
        
        x = self.layer0(self.from_rgb1(x_fs))
        # x = self.layer1(torch.cat((x, self.from_rgb2(x_ds2)), 1))
        # x = self.layer2(torch.cat((x, self.from_rgb3(x_ds4)), 1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        

class MSG_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.lrelu = nn.LeakyReLU(0.2)

        self.from_rgb1 = nn.Sequential(
            nn.Conv2d(3, 16, (1, 1), bias=True))
        self.from_rgb2 = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1), bias=True))
        self.from_rgb3 = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1), bias=True))

        self.layer0 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(16, 32, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(35, 32, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32, 64, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(67, 64, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2), 
            nn.AvgPool2d(2, stride=2, padding=0)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, (8, 8), padding=0, bias=True),
            nn.LeakyReLU(0.2)
            )
        self.fc = nn.Linear(128, 1, bias=True)


    def forward(self, x):
        """
        forward pass of the discriminator
        :param inputs: dict of multiscale input images
        :return raw prediction values
        """
        assert type(x) == dict, "inputs must be a dictionary"
        x_fs, x_ds2, x_ds4 = x['fs'], x['ds2'], x['ds4']

        x = self.layer0(self.from_rgb1(x_fs))
        x = self.layer1(torch.cat((x, self.from_rgb2(x_ds2)), 1))
        x = self.layer2(torch.cat((x, self.from_rgb3(x_ds4)), 1))
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x



if __name__ == '__main__':
    model = EnvAutoEncoder(vector_size=1024, sphere_size=64)

    print(model)
    # x = {'pred': torch.randn(16, 3, 64, 64), 'pred_ds2': torch.randn(16, 3, 32, 32), 'pred_ds4': torch.randn(16, 3, 16, 16)}
    # out = model(x['pred'], x['pred_ds2'], x['pred_ds4'])
    x = torch.randn(16, 3, 64, 64)
    o1, o2, o3 = model(x)
    print(o1.shape, o2.shape, o3.shape)