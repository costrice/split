
import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models as models


class Encoder(nn.Module):
    """
    Encode an RGB image into 1D latent code.
    """
    def __init__(self, out_dim):
        super().__init__()
        self.latent_code_dim = out_dim
        self.encoder = models.resnet18(pretrained=True)
        self.num_feature = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(self.num_feature, self.latent_code_dim)

    def forward(self, x):
        latent_code = self.encoder(x)
        return latent_code


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
        

class Decoder(nn.Module):
    """
    Decode a 1D latent code into an RGB image.
    """
    def __init__(self, input_height=256, num_Blocks=[2, 2, 2, 2], latent_code_dim=32, nc=3):
        super().__init__()
        self.out_height = input_height
        self.in_planes = 512

        self.linear = nn.Linear(latent_code_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(int(self.out_height) // 2, int(self.out_height) // 2), mode="bilinear")
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, int(self.out_height), int(self.out_height))
        return x
        

class AutoEncoder(nn.Module):
    """
    A common auto-encoder.
    """
    def __init__(self, input_hw, latent_dim):
        super().__init__()

        self.input_hw = input_hw
        self.latent_dim = latent_dim

        self.encoder = Encoder(out_dim=self.latent_dim)
        self.decoder = Decoder(latent_code_dim=self.latent_dim, input_height=self.input_hw)
        
    def forward(self, batch):
        albedo_texture = batch["albedo_texture"]
        mask = batch["mask"]
        latent_code = self.encoder(albedo_texture * mask)
        recon = self.decoder(latent_code)
        return {
            "code": latent_code,
            "recon": recon,
        }


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         # [b, 784] => [b, 20]
#         # u: [b, 10]
#         # sigma: [b, 10]
#         self.encoder = nn.Sequential(
#             # [b, 784] => [b, 256]
#             nn.Linear(784, 256),
#             nn.ReLU(),
#             # [b, 256] => [b, 64]
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             # [b, 64] => [b, 20]
#             nn.Linear(64, 20),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             # [b, 10] => [b, 64]
#             nn.Linear(10, 64),
#             nn.ReLU(),
#             # [b, 64] => [b, 256]
#             nn.Linear(64, 256),
#             nn.ReLU(),
#             # [b, 256] => [b, 784]
#             nn.Linear(256, 784),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, x):
#         """
#         :param [b, 1, 28, 28]:
#         :return [b, 1, 28, 28]:
#         """
#         batchsz = x.size(0)
#         # flatten
#         x = x.view(batchsz, -1)
#         # encoder
#         # [b, 20] including mean and sigma
#         q = self.encoder(x)
#         # [b, 20] => [b, 10] and [b, 10]
#         mu, sigma = q.chunk(2, dim=1)
#         # reparameterize trick,  epsilon~N(0, 1)
#         q = mu + sigma * torch.randn_like(sigma)
#
#
#         # decoder
#         x_hat = self.decoder(q)
#         # reshape
#         x_hat = x_hat.view(batchsz, 1, 28, 28)
#
#
#         # KL
#         kld = 0.5 * torch.sum(
#             torch.pow(mu, 2) +
#             torch.pow(sigma, 2) -
#             torch.log(1e-8 + torch.pow(sigma, 2)) - 1
#         ) / (batchsz*28*28)
#
#
#         return x_hat, kld
 
 
# class AutoEncoder(nn.Module):
#     def __init__(self, input_hw, enc_type, latent_dim):
#         super().__init__(input_hw=input_hw, enc_type=enc_type, latent_dim=latent_dim)

#         self.input_hw = input_hw
#         self.latent_dim = latent_dim
#         self.enc_out_dim = 512

#         self.encoder = resnet18_encoder()
#         self.decoder_mid = resnet18_decoder(self.latent_dim, self.input_hw)
#         self.decoder_small = resnet18_decoder(self.latent_dim, self.input_hw)

#         self.fc_mid = nn.Linear(self.enc_out_dim, self.latent_dim)
#         self.fc_small = nn.Linear(self.enc_out_dim, self.latent_dim)

#         self.automatic_optimization = False

#         self.lr = 0.001
        

#     def forward(self, batch):
#         feats = self.encoder(batch)
#         z_mid = self.fc_mid(feats)
#         z_small = self.fc_small(feats)
        

#         x_hat_mid = self.decoder_mid(z_mid)
#         x_hat_small = self.decoder_small(z_small)

#         x_hat = x_hat_mid + x_hat_small
#         return x_hat

#     def step(self, batch, batch_idx):
#         batch = batch

#         feats = self.encoder(batch)
#         z_mid = self.fc_mid(feats)
#         z_small = self.fc_small(feats)

#         x_hat_mid = self.decoder_mid(z_mid)
#         x_hat_small = self.decoder_small(z_small)

#         x_hat = x_hat_mid + x_hat_small

#         # InfoLoss(z_mid, x_hat_mid) 
#         # InfoLoss(z_small, x_hat_small)
#         # SparsityLoss(z_small)




#     def configure_optimizers(self):
#         opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
#         opt_dec_mid = torch.optim.Adam(self.decoder_mid.parameters(), lr=self.lr)
#         opt_dec_small = torch.optim.Adam(self.decoder_small.parameters(), lr=self.lr)
#         return opt_enc, opt_dec_mid, opt_dec_small