# -*- coding: utf-8 -*-
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blurpool import BlurPool

# type alias
Sample = Dict[str, Union[torch.Tensor, List]]


class _DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, blur_pool=True):
        super().__init__()
        # Pooling
        if blur_pool:
            self.down = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1), BlurPool(in_channels, stride=2)
            )
        else:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolution
        self.conv = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            )
        ]
        if use_bn:
            self.conv.append(nn.BatchNorm2d(num_features=out_channels))
        self.conv.append(nn.SiLU(inplace=True))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class _UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        # Up-sampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution
        self.conv = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
            )
        ]
        if use_bn:
            self.conv.append(nn.BatchNorm2d(num_features=out_channels))
        self.conv.append(nn.SiLU(inplace=True))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x, skipped):
        x = self.up(x)
        x = torch.cat([x, skipped], dim=1)
        return self.conv(x)


def _initialize_weights(net: nn.Module):
    """
    Copied from effnetv2.py.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            m.bias.data.zero_()


class _EncoderDecoderSkip(nn.Module):
    """
    Structure modified from "Total Relighting: Learning to Relight Portraits for
    Background Replacement". A U-net architecture with 13 encoder-decoder down
    and skip connections.
    Each layer is run through 3 x 3 convolutions followed by BatchNorm and SiLU
    activation. The number of filters are 32, 64, 128, 256, 512, 512 for the
    encoder, 512 for the bottleneck, and 256, 128, 64, 32, 32, 32 for the
    decoder respectively.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        base_filters=32,
        color_vector_dim=0,
        use_bn=True,
        blur_pool=True,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.color_vector_dim = color_vector_dim
        self.use_batch_norm = use_bn
        self.use_blur_pooling = blur_pool
        self.dropout_rate = dropout_rate

        # filter amount settings
        in_conv_filters = base_filters * 1
        encoder_filters = [
            base_filters * 1,
            base_filters * 2,
            base_filters * 4,
            base_filters * 8,
            base_filters * 16,
            base_filters * 16,
        ]
        bottleneck_filters = base_filters * 16
        decoder_filters = [
            base_filters * 16,
            base_filters * 16,
            base_filters * 8,
            base_filters * 4,
            base_filters * 2,
            base_filters * 1,
        ]

        assert len(encoder_filters) == len(decoder_filters)
        n_layers = len(encoder_filters)

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_conv_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_conv_filters),
            nn.SiLU(inplace=True),
        )

        self.encoder = nn.ModuleList(
            [
                _DownConv(
                    in_conv_filters,
                    encoder_filters[0],
                    use_bn=use_bn,
                    blur_pool=blur_pool,
                )
            ]
        )

        for index in range(1, n_layers):
            self.encoder.append(
                _DownConv(
                    encoder_filters[index - 1],
                    encoder_filters[index],
                    use_bn=use_bn,
                    blur_pool=blur_pool,
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                encoder_filters[-1], bottleneck_filters, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(bottleneck_filters),
            nn.SiLU(inplace=True),
        )

        # count skip-connection in in_channels
        self.decoder = nn.ModuleList(
            [
                _UpConv(
                    bottleneck_filters + encoder_filters[-2],
                    decoder_filters[0],
                    use_bn=use_bn,
                )
            ]
        )

        if color_vector_dim:
            # if color_vector_dim != 0, use Color-Feature Modifier
            self.modifier_mul = nn.ModuleList(
                [nn.Linear(color_vector_dim, decoder_filters[0])]
            )
            self.modifier_add = nn.ModuleList(
                [nn.Linear(color_vector_dim, decoder_filters[0])]
            )

        for index in range(1, n_layers - 1):
            self.decoder.append(
                _UpConv(
                    decoder_filters[index - 1] + encoder_filters[-2 - index],
                    decoder_filters[index],
                    use_bn=use_bn,
                )
            )
            if color_vector_dim:
                self.modifier_mul.append(
                    nn.Linear(color_vector_dim, decoder_filters[index])
                )
                self.modifier_add.append(
                    nn.Linear(color_vector_dim, decoder_filters[index])
                )

        self.decoder.append(
            _UpConv(
                decoder_filters[-2] + in_conv_filters,
                decoder_filters[-1],
                use_bn=use_bn,
            )
        )

        self.out_conv = nn.Conv2d(decoder_filters[-1], out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout_rate)

        _initialize_weights(self)

    def forward(self, x: torch.Tensor, color_feature=None):
        x = self.in_conv(x)
        down_features = [x]
        # encode
        for layer in self.encoder:
            x = layer(x)
            x = self.dropout(x)
            down_features.append(x)
        # bottleneck
        x = self.bottleneck(down_features[-1])
        x = self.dropout(x)
        # decode
        for index, layer in enumerate(self.decoder):
            x = layer(x, down_features[-2 - index])
            x = self.dropout(x)
            # skip the last decoded feature
            if self.color_vector_dim and index < len(self.decoder) - 1:
                if color_feature is not None:
                    mul = self.modifier_mul[index](color_feature)[:, :, None, None]
                    add = self.modifier_add[index](color_feature)[:, :, None, None]
                    x = x * mul + add  # channel-wise multiply and add
                else:
                    raise ValueError("Error: Color feature vector is not " "provided.")
        # 1x1 conv
        x = self.out_conv(x)
        return x


class I2ANNet(nn.Module):
    """
    Normal-Albedo Net:
        Input: Face, Mask
        Output: Normal in [-1, 1], Albedo
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=4,
            out_channels=6,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group: Sample):
        face = input_group["face"]
        mask = input_group["mask"]
        output = self.backbone(torch.cat([face * mask, mask], dim=1))
        # normalize normal to [-1, 1]
        normal = output[:, :3]
        normal = normal / (normal ** 2).sum(dim=1, keepdim=True).add(1e-12).sqrt()
        # get albedo in [0, 1]
        albedo = torch.sigmoid(output[:, 3:6])
        return {"normal": normal * mask, "albedo": albedo * mask}


class I2ANet(nn.Module):
    """
    Albedo Net:
        Input: Face, Mask
        Output: Albedo
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=4,
            out_channels=3,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group):
        face = input_group["face"]
        mask = input_group["mask"]
        output = self.backbone(torch.cat([face * mask, mask], dim=1))
        # get albedo in [0, 1]
        albedo = torch.sigmoid(output)
        return {"albedo": albedo * mask}


class I2NNet(nn.Module):
    """
    Normal Net:
        Input: face, mask
        Output: normal in [-1, 1]
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=4,
            out_channels=3,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group):
        face = input_group["face"]
        mask = input_group["mask"]
        normal = self.backbone(torch.cat([face * mask, mask], dim=1))
        # normalize to [-1, 1]
        normal = normal / (normal ** 2).sum(dim=1, keepdim=True).add(1e-12).sqrt()
        return {"normal": normal * mask}


class IAN2DSNet(nn.Module):
    """
    Shading-Specular Net:
        Input: face, normal in [-1, 1], albedo, mask
        Output: diffuse shading, specular
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=10,
            out_channels=6,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group: Sample):
        face = input_group["face"]
        normal = input_group["normal"]
        albedo = input_group["albedo"]
        mask = input_group["mask"]  # binary
        output = self.backbone(
            torch.cat([face * mask, normal * mask, albedo * mask, mask], dim=1)
        )

        specular = F.softplus(output[:, :3])  # map to [0, +inf]
        diffuse_shading = F.softplus(output[:, 3:])  # map to [0, +inf]

        recon = torch.clip(diffuse_shading * albedo + specular, 0, 1)
        return {
            "shading": diffuse_shading * mask,
            "specular": specular * mask,
            "face": recon * mask,
        }


class I2DSNet(nn.Module):
    """
    Shading-Specular Net:
        Input: face, mask
        Output: diffuse shading, specular
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=4,
            out_channels=6,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group):
        face = input_group["face"]
        albedo = input_group["albedo"]
        mask = input_group["mask"]
        output = self.backbone(torch.cat([face * mask, mask], dim=1))

        specular = F.softplus(output[:, :3])  # map to [0, +inf]
        diffuse_shading = F.softplus(output[:, 3:])  # map to [0, +inf]

        recon = torch.clip(diffuse_shading * albedo + specular, 0, 1)
        return {
            "shading": diffuse_shading * mask,
            "specular": specular * mask,
            "face": recon * mask,
        }


class IA2DSNet(nn.Module):
    """
    Shading-Specular Net:
        Input: face, albedo, mask
        Output: diffuse shading, specular
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=7,
            out_channels=6,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group):
        face = input_group["face"]
        albedo = input_group["albedo"]
        mask = input_group["mask"]
        output = self.backbone(torch.cat([face * mask, albedo * mask, mask], dim=1))

        specular = F.softplus(output[:, :3])  # map to [0, +inf]
        diffuse_shading = F.softplus(output[:, 3:])  # map to [0, +inf]

        recon = torch.clip(diffuse_shading * albedo + specular, 0, 1)
        return {
            "shading": diffuse_shading * mask,
            "specular": specular * mask,
            "face": recon * mask,
        }


class IN2DSNet(nn.Module):
    """
    Shading-Specular Net:
        Input: face, normal in [-1, 1], mask
        Output: diffuse shading, specular
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=7,
            out_channels=6,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group):
        face = input_group["face"]
        normal = input_group["normal"]
        albedo = input_group["albedo"]
        mask = input_group["mask"]  # binary
        output = self.backbone(torch.cat([face * mask, normal * mask, mask], dim=1))

        specular = F.softplus(output[:, :3])  # map to [0, +inf]
        diffuse_shading = F.softplus(output[:, 3:])  # map to [0, +inf]

        recon = torch.clip(diffuse_shading * albedo + specular, 0, 1)
        return {
            "shading": diffuse_shading * mask,
            "specular": specular * mask,
            "face": recon * mask,
        }


class I2ANDSNet(nn.Module):
    """
    Direct Net:
        Input: face, mask
        Output: normal in [-1, 1], albedo, shading, specular
        Architecture: U-net with skip-connection
    """

    def __init__(
        self, base_filters: int = 32, use_bn: bool = True, use_blur_pool: bool = True
    ):
        super().__init__()
        self.backbone = _EncoderDecoderSkip(
            in_channels=4,
            out_channels=12,
            base_filters=base_filters,
            use_bn=use_bn,
            blur_pool=use_blur_pool,
        )

    def forward(self, input_group: Sample):
        face = input_group["face"]
        mask = input_group["mask"]
        output = self.backbone(torch.cat([face * mask, mask], dim=1))
        # normalize normal to [-1, 1]
        normal = output[:, :3]
        normal = normal / (normal ** 2).sum(dim=1, keepdim=True).add(1e-12).sqrt()
        # get albedo in [0, 1]
        albedo = torch.sigmoid(output[:, 3:6])
        # get light component
        specular = F.softplus(output[:, 6:9])  # map to [0, +inf]
        diffuse_shading = F.softplus(output[:, 9:])  # map to [0, +inf]
        # compute recon
        diffuse = diffuse_shading * albedo
        recon = diffuse + specular
        return {
            "normal": normal * mask,
            "albedo": albedo * mask,
            "shading": diffuse_shading * mask,
            "specular": specular * mask,
            "face": recon * mask,
        }


class CascadeNetwork(nn.Module):
    """
    Combine two models together.
    """

    def __init__(self, model0: nn.Module, dev0: int, model1: nn.Module, dev1: int):
        """
        Put different model on different cards.
        """
        super().__init__()
        # set device
        self.devm = torch.device(dev0)
        self.dev0 = torch.device(dev0)
        self.dev1 = torch.device(dev1)

        # create models
        self.model0 = model0.to(device=self.dev0)
        self.model1 = model1.to(device=self.dev1)

    def forward(self, input_group: Sample):
        face = input_group["face"]
        mask = input_group["mask"]
        # construct input of model0
        input0 = {"face": face.to(self.dev0), "mask": mask.to(self.dev0)}
        # get prediction of model0
        pred0 = self.model0(input0)
        # construct input of model1
        input1 = {"face": face.to(self.dev1), "mask": mask.to(self.dev1)}
        for compo, img in pred0.items():
            input1[compo] = img.to(self.dev1)
        # get prediction of model1
        pred1 = self.model1(input1)
        # combine predictions and put onto dev0
        for compo, img in pred0.items():
            pred0[compo] = img.to(self.devm)
        for compo, img in pred1.items():
            pred0[compo] = img.to(self.devm)
        return pred0
