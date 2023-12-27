import math

import torch
import torch.nn.functional as F

# import loss

eps = 1e-6

import numpy as np
import torch.nn as nn

import lpips
from models.spherical_lighting_estimator import tonemapping

from utils.warp import warp_sp2rect



import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Dict
# from utils.ibrender import IBRenderer
from utils.general import read_image, write_image, linrgb2srgb, \
    srgb2linrgb, tensor2ndarray

def detach_and_cpu(tensor):
    """
    Detach and move a tensor to CPU.
    tensor: torch.Tensor
    """
    return tensor.detach().cpu()



# def L1Accuracy_sp(pred, target):
#     """
#     Calculate the L1 accuracy of a batch of predictions and targets.
#     pred: LOG SPACE torch.Tensor of shape (batch_size, SIZE_OF_SPHERE)
#     target: torch.Tensor of shape (batch_size, SIZE_OF_SPHERE)
#     """
#     pred = detach_and_cpu(pred)
#     target = detach_and_cpu(target)

#     # pred_exp = torch.exp(pred) - 1

#     # Calculate the mean L1 distance in non-zero elements
#     # non_zero_elements = target != 0

#     # generate row and column index
#     hw = pred.shape[2]
#     idx = np.arange(hw)
#     idx_row = np.repeat(idx[:, None], repeats=hw, axis=1)
#     idx_col = np.repeat(idx[None, :], repeats=hw, axis=0)
#     mask = (((idx_row - hw / 2) ** 2 + (idx_col - hw / 2) ** 2) < (hw / 2) ** 2).astype(np.float32)
#     # write_image('mask.png', mask)
#     # mask = torch.from_numpy(mask)

#     # TODO mask to sphere?
#     # l1_dist = nn.L1Loss(reduction='sum')(pred_exp, target)
#     l1_dist = nn.L1Loss(reduction='sum')(pred * mask, target * mask)
#     # torch.sum(torch.abs(pred_exp - target), dim=1)

#     mask = torch.from_numpy(mask)
#     # st()
#     l1_dist = l1_dist / torch.sum(mask) / target.shape[0] / target.shape[1]
#     # l1_dist = l1_dist / torch.sum(non_zero_elements, dim=1)
#     # l1_dist = torch.mean(l1_dist[non_zero_elements])

#     return l1_dist


class MaskedMeanLpLoss(object):
    """
    Compute the mean of power to the p of differences between masked label and prediction.
    """
    def __init__(self,
                 p: float):
        """Set the exponent of power.
        
        Args:
            p: the exponent of power
        """
        self.p = p
    
    def __call__(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor):
        """
        Computes the L_p Loss for the input two images `pred` and `gt` within the mask.

        Args:
            pred: [b, c, h, w] an image batch,
            gt: [b, c, h, w] another image batch,
            mask: [b, 1, h, w], the image mask within which error is computed.

        Returns:
            torch.Tensor: scalar, average L-p loss, smaller the better
        """
        gt = gt.clip(0, 1)
        pred = pred.clip(0, 1)
        
        diff = torch.abs(((gt - pred) * mask)) ** self.p
        spatial_average = torch.sum(diff, dim=(2, 3)) / torch.sum(mask, dim=(2, 3))
        return spatial_average.mean()


class MaskedRMSECalculator(object):
    """
    Computes Root Mean-Square Error of two input images (one GT, one Pred).
    """
    def __init__(self):
        self.mse_fn = MaskedMeanLpLoss(p=2)
    
    def __call__(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor):
        """Computes PSNR for the input two images `pred` and `gt` within the mask.
        
        Args:
            pred: [b, c, h, w] an image batch,
            gt: [b, c, h, w] another image batch,
            mask: [b, 1, h, w], the image mask within which error is computed.

        Returns:
            torch.Tensor: scalar, average RMSE, smaller the better
        """
        
        rmse = self.mse_fn(pred, gt, mask) ** 0.5
        return rmse


class MaskedPSNRCalculator(object):
    """
    Computes Peak Signal-Noise Ratio of two input images (one GT, one Pred).
    """
    def __init__(self):
        self.mse_fn = MaskedMeanLpLoss(p=2)
    
    def __call__(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor):
        """Computes PSNR for the input two images `pred` and `gt` within the mask.
        
        Args:
            pred: [b, c, h, w] an image batch,
            gt: [b, c, h, w] another image batch,
            mask: [b, 1, h, w], the image mask within which error is computed.

        Returns:
            torch.Tensor: scalar, average PSNR, bigger the better
        """
        # gt = gt.clip(0, 1)
        # pred = pred.clip(0, 1)
        
        mse = self.mse_fn(pred, gt, mask)
        return 10 * torch.log10(1 / mse)


class SSIMCalculator(object):
    """modified from https://github.com/jorge-pessoa/pytorch-msssim"""
    
    def __init__(self):
        pass
    
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(w_size)])
        return gauss / gauss.sum()
    
    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window
    
    def __call__(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 w_size=11,
                 size_average=True,
                 full=False):
        """Computes SSIM for the two input images `y_true` and `y_pred`.
        
        Args:
            y_true: 4-d Tensor in [batch_size, channels, img_rows, img_cols]
            y_pred: 4-d Tensor in [batch_size, channels, img_rows, img_cols]
            w_size: int, default 11
            size_average: boolean, default True
            full: boolean, default False
            
        Return:
            torch.Tensor: scalar, SSIM, larger the better.
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        
        # gt = gt.clip(0, 1)
        # pred = pred.clip(0, 1)
        
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
        
        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)
        
        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2
        
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
        
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        
        if full:
            return ret, cs
        return ret


class MaskedSimilarityCalculator(object):
    """
    Simultaneously computes RMSE, PSNR and SSIM for two input images (one GT, one Pred).
    """
    
    def __init__(self):
        self.RMSE_fn = MaskedRMSECalculator()
        self.PSNR_fn = MaskedPSNRCalculator()
        self.SSIM_fn = SSIMCalculator()
    
    def __call__(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor):
        """
        Computes RMSE, PSNR and SSIM for the two input images, considering mask.
        
        Args:
            pred: [b, c, h, w] an image batch,
            gt: [b, c, h, w] another image batch,
            mask: [b, 1, h, w], the image mask within which error is computed.
            
        Returns:
            RMSE, PSNR, SSIM, each torch.Tensor scalar.
        """
        RMSE = self.RMSE_fn(pred, gt, mask)
        PSNR = self.PSNR_fn(pred, gt, mask)
        SSIM = self.SSIM_fn(pred * mask, gt * mask)
        return {"RMSE": RMSE, "PSNR": PSNR, "SSIM": SSIM}



class MaskedAngularError(object):
    """
    Modified from matlab : colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)));
    angle = 180 / pi * angle;
    """
    
    def __init__(self, des='average Angular Error'):
        self.des = des
    
    def __repr__(self):
        return "Masked Angular Error"
    
    def __call__(self, pred, gt, mask):
        """
        args:
            pred : 4-d tensor in [b, c, h, w]
            gt : 4-d tensor in [b, c, h, w]
            mask: 4-d tensor in [b, 1, h, w]
        return mean angular error, smaller the better
        """
        dotP = torch.sum(pred * gt * mask, dim=1)
        norm_pred = torch.sqrt(torch.sum(pred * pred * mask, dim=1))
        norm_gt = torch.sqrt(torch.sum(gt * gt * mask, dim=1))
        ae = 180 / math.pi * torch.acos(dotP / (norm_pred * norm_gt + eps))
        ae = torch.sum(ae * mask[:, 0], dim=(1, 2)) / torch.sum(mask[:, 0], dim=(1, 2))
        return ae.mean()
    

class MetricNormal(object):
    def __init__(self):
        self.angular_error_fn = MaskedAngularError()
    
    def __call__(self, pred, gt):
        """
        Args:
            pred: predicted components as a dict.
                Must contain "normal".
            gt: input (or gt) components as a dict.
                Must contain "normal", "mask".
        Return:
            Computed error metric as a dict.
        """
        # transform normal from [0, 1] to [-1, 1]
        angular_eror = self.angular_error_fn(pred["normal"] * 2 - 1,
                                             gt["normal"] * 2 - 1,
                                             gt["mask"])
        return {
            "angular_error": angular_eror
        }


class MetricDelightNet(object):
    """
    Error metric for model DelightNet.
    """
    
    def __init__(self):
        self.similarity_metrics = MaskedSimilarityCalculator()
    
    def __call__(self, pred, gt):
        """
        Args:
            pred: predicted components as a dict.
                Must contain "albedo", "shading", "specular".
            gt: input (or gt) components as a dict.
                Must contain "albedo", "shading", "specular", "mask"
        Return:
            Computed error metric as a dict.
        """
        albedo_rmse, albedo_psnr, albedo_ssim = \
            self.similarity_metrics(pred["albedo"], gt["albedo"], gt["mask"])
        specular_rmse, specular_psnr, specular_ssim = \
            self.similarity_metrics(pred["specular"], gt["specular"], gt["mask"])
        shading_gt_max = torch.amax(gt["shading"], dim=(1, 2, 3), keepdim=True)
        shading_rmse, shading_psnr, shading_ssim = \
            self.similarity_metrics(pred["shading"] / shading_gt_max,
                                    gt["shading"] / shading_gt_max,
                                    gt["mask"])
        
        return {
            "albedo_rmse": albedo_rmse,
            "albedo_psnr": albedo_psnr,
            "albedo_ssim": albedo_ssim,
            "shading_rmse": shading_rmse,
            "shading_psnr": shading_psnr,
            "shading_ssim": shading_ssim,
            "specular_rmse": specular_rmse,
            "specular_psnr": specular_psnr,
            "specular_ssim": specular_ssim,
        }


class MetricDecomposer(object):
    """
    Error metric for model DelightNetFullPipeline.
    """
    
    def __init__(self):
        self.similarity_metrics = MaskedSimilarityCalculator()
        self.angular_error_fn = MaskedAngularError()
    
    def __call__(self, pred, gt):
        """
        Args:
            pred: predicted components as a dict.
                Must contain "albedo_licol", "shading", "specular".
            gt: input (or gt) components as a dict.
                Must contain "albedo_licol", "shading", "specular", "mask"
        Return:
            Computed error metric as a dict.
        """
        # transform normal from [0, 1] to [-1, 1]
        normal_angular_error = self.angular_error_fn(pred["normal"] * 2 - 1,
                                                     gt["normal"] * 2 - 1,
                                                     gt["mask"])

        albedo_rmse, albedo_psnr, albedo_ssim = \
            self.similarity_metrics(pred["albedo"], gt["albedo"], gt["mask"])

        shading_gt_max = torch.amax(gt["shading"], dim=(1, 2, 3), keepdim=True)
        shading_rmse, shading_psnr, shading_ssim = \
            self.similarity_metrics(pred["shading"] / shading_gt_max,
                                    gt["shading"] / shading_gt_max,
                                    gt["mask"])

        specular_rmse, specular_psnr, specular_ssim = \
            self.similarity_metrics(pred["specular"], gt["specular"], gt["mask"])
        
        return {
            "normal_angular_error": normal_angular_error,
            "albedo_rmse": albedo_rmse,
            "albedo_psnr": albedo_psnr,
            "albedo_ssim": albedo_ssim,
            "shading_rmse": shading_rmse,
            "shading_psnr": shading_psnr,
            "shading_ssim": shading_ssim,
            "specular_rmse": specular_rmse,
            "specular_psnr": specular_psnr,
            "specular_ssim": specular_ssim,
        }

class MetricLight(object):
    """
    Simultaneously computes RMSE, PSNR, SSIM and LPIPS for two input images (one GT, one Pred).
    """
    
    def __init__(self, device):
        self.RMSE_fn = MaskedRMSECalculator()
        self.PSNR_fn = MaskedPSNRCalculator()
        self.SSIM_fn = SSIMCalculator()
        self.lpips_net = lpips.LPIPS(net="vgg").to(device)
        self.lpips_fn = lambda x, y: self.lpips_net(x, y, normalize=True).mean()
        self.device = device
    
    def __call__(self,
                 pred: torch.Tensor,
                 gt: torch.Tensor,
                 mask: torch.Tensor):
        """
        Computes RMSE, PSNR and SSIM for the two input images, considering mask.
        
        Args:
            pred: [b, c, h, w] an image batch,
            gt: [b, c, h, w] another image batch,
            mask: [b, 1, h, w], the image mask within which error is computed.
            
        Returns:
            RMSE, PSNR, SSIM, each torch.Tensor scalar.
        """
        pred = pred.clip(0, 1)
        gt = gt.clip(0, 1)
        # breakpoint()
        
        RMSE = self.RMSE_fn(pred, gt, mask)
        PSNR = self.PSNR_fn(pred, gt, mask)
        SSIM = self.SSIM_fn(tonemapping(pred * mask), tonemapping(gt * mask))
        if pred.device == 'cpu':
            pred = pred.to(self.device)
            gt = gt.to(self.device)
            mask = mask.to(self.device)
        LPIPS = self.lpips_fn(tonemapping(pred * mask), tonemapping(gt * mask))
        
        return {"RMSE": RMSE, "PSNR": PSNR, "SSIM": SSIM, "LPIPS": LPIPS}

    def get_angular_error(self, 
                          pred: torch.Tensor,
                          gt: torch.Tensor,
                          mask: torch.Tensor, 
                          get_abs=False, 
                          cloudy_mask=None,
                          ):
        batch_size = pred.shape[0]
        angular = []
        pred = pred * mask
        
        for i in range(batch_size):
            pred_i = pred[i].permute(1, 2, 0).cpu().numpy()
            gt_i = gt[i].permute(1, 2, 0).cpu().numpy()
            
            ang = get_angular_metrics(pred_i, gt_i, get_abs=get_abs)
            # breakpoint()
            # az_ls.append(az)
            # el_ls.append(el)
            # breakpoint()
            if cloudy_mask is None or not cloudy_mask[i].item(): # not cloudy
                angular.append(ang)
            # print(ang)
            if ang != ang: 
                breakpoint()
        # az = np.mean(az_ls)
        # el = np.mean(el_ls)
        angular = np.mean(angular)
        return angular, angular

def get_angular_metrics(pred, gt, get_abs=True):
    pred_dir = get_angular_metrics_single(pred)
    gt_dir = get_angular_metrics_single(gt)
    # breakpoint()

    # compute angle
    cos_angle = np.dot(pred_dir, gt_dir)
    cos_angle = np.clip(cos_angle, -1, 1)
    if cos_angle > 1:
        print("cos_angle > 1")
        print(cos_angle)
        print(pred_dir)
        print(gt_dir)
    angle = np.arccos(cos_angle) / np.pi * 180

    # if get_abs:
    #     angle = np.abs(angle)
    return angle

def get_angular_metrics_single(envmap_sp: np.ndarray):
    """
    Return the direction of the brightest pixel in a spherical envmap.
    Args:
        envmap_sp (np.ndarray): [h, w, 3], spherical envmap.
    Returns:
        [x, y, z]: the direction of the brightest pixel.
    """
    # One way: directly use spherical form
    envmap_sp_mean = np.mean(envmap_sp, axis=2)
    y, x = np.unravel_index(np.argmax(envmap_sp_mean), envmap_sp_mean.shape)
    # convert to [-1, 1]
    x = (x - envmap_sp.shape[1] / 2) / (envmap_sp.shape[1] / 2)
    y = -(y - envmap_sp.shape[0] / 2) / (envmap_sp.shape[0] / 2)
    offset = 1 - x ** 2 - y ** 2
    offset = np.clip(offset, 0, None)
    z = np.sqrt(offset)
    normal = np.array([x, y, z], dtype=np.float32)
    # compute reflection vector
    view = np.array([0, 0, 1], dtype=np.float32)
    refl = normal * normal[2] * 2 - view  # 2(v dot n)n - v

    return refl
       
def get_rerender_error(batch, lighting, ibr):

    recon = re_render_face(batch, lighting, ibr)
    RMSE = MaskedRMSECalculator()(recon, batch['rerender_input'], batch['rerender_mask'])
    return RMSE

def re_render_face(
        face_batch,
        lighting,
        ibrenderer,
        adjust_recon_mean: bool = True,
):
    """
    Re-render the input face image, using the estimated normal, albedo, mask,
    and lighting.
    Args:
        face_batch (Dict): keys should include 'normal' in [-1, 1], 'albedo'.
            If adjust_recon_mean is True, keys should also include 'input' and
            'mask'.
        lighting (torch.Tensor): the estimated sphere lighting.
        ibrenderer (IBRenderer): the IBRenderer object.
        adjust_recon_mean (bool): whether to adjust the intensity of the
            reconstructed face to match the input face. Default: True.
    Returns:
        torch.Tensor: the re-rendered face image.
    """
    # first, image-based render the estimated lighting to get diffuse shading
    # and specular component

    ibrendered = ibrenderer.render_using_mirror_sphere(lighting)
    shading_sp = ibrendered['diffuse']
    specular_sp = ibrendered['specular']

    # write_image(output_dir / 'shading_sp.png', linrgb2srgb(
    #     tensor2ndarray(shading_sp)))
    # write_image(output_dir / 'specular_sp.png', linrgb2srgb(
    #     tensor2ndarray(specular_sp)))

    # project sphere map to face according to normal
    # first, convert normal to coordinate on the sphere
    sphere_hw = shading_sp.shape[-1]
    normal = face_batch['rerender_normal']
    w_idx = normal[:, 0, :, :]  # (b, h_face, w_face) in [-1, 1]
    h_idx = -normal[:, 1, :, :]  # (b, h_face, w_face) in [-1, 1]
    # get the corresponding pixel value on the sphere using grid_sample
    grid = torch.stack([w_idx, h_idx], dim=-1)  # (b, h_face, w_face, 2)
    shading = F.grid_sample(shading_sp, grid, align_corners=True)
    specular = F.grid_sample(specular_sp, grid, align_corners=True)

    # write_image(output_dir / 'shading.png', linrgb2srgb(
    #     tensor2ndarray(shading)))
    # write_image(output_dir / 'specular.png', linrgb2srgb(
    #     tensor2ndarray(specular)))

    # reconstruct the input face using shading, albedo, and specular
    albedo = face_batch['rerender_albedo']
    recon = shading * albedo + specular

    # write_image(output_dir / 'recon.png', linrgb2srgb(
    #     tensor2ndarray(recon)))

    if adjust_recon_mean:
        # adjust the intensity of the reconstructed face to match the input face
        input_face = face_batch['rerender_input']
        mask = face_batch['rerender_mask']
        if mask.size(1) == 1:
            mask = mask.repeat(1, 3, 1, 1)
        input_mean = torch.sum(input_face * mask, dim=(1, 2, 3), keepdim=True) \
                     / torch.sum(mask, dim=(1, 2, 3), keepdim=True)  # (b, 1, 1, 1)
        recon_mean = torch.sum(recon * mask, dim=(1, 2, 3), keepdim=True) \
                        / torch.sum(mask, dim=(1, 2, 3), keepdim=True)  # (b, 1, 1, 1)
        recon = recon * (input_mean / (recon_mean + 1e-8))

        # write_image(output_dir / 'input.png', linrgb2srgb(
        #     tensor2ndarray(input_face)))
        # write_image(output_dir / 'recon_adjusted.png', linrgb2srgb(
        #     tensor2ndarray(recon)))

    return recon
       
       


# def get_angular_metircs(pred, gt, get_abs=False):
#     pred_a, pred_e = get_angular_metircs_sing(pred)
#     gt_a, gt_e = get_angular_metircs_sing(gt)
#     # breakpoint() 
    
#     az_error = np.abs(pred_a - gt_a)
#     if az_error >= 180:
#         if gt_a - pred_a > 0:
#             az_error = (gt_a - pred_a) - 360
#         else:
#             az_error = 360 + (gt_a - pred_a)
#     else:
#         az_error = gt_a - pred_a
#     ele_error = gt_e - pred_e
    
#     if get_abs:
#         az_error = np.abs(az_error)
#         ele_error = np.abs(ele_error)
#     # az_error = az_error * np.pi / 180
#     # ele_error = ele_error * np.pi / 180
#     return az_error, ele_error

# def get_angular_metircs_sing(envmap):
#     envmap_rect = warp_sp2rect(envmap, out_h=64)
#     max_pos = np.unravel_index(np.argmax(envmap_rect), envmap_rect.shape)
#     max_pos = max_pos[:2]
#     azimuth = max_pos[1] / envmap_rect.shape[1] * 360
#     elevation = max_pos[0] / envmap_rect.shape[0] * 180
#     return azimuth, elevation



if __name__ == "__main__":
    b, c, h, w = 4, 3, 256, 256
    vec_a = torch.tensor((0.99, 0, 0), dtype=torch.float32).view(-1, 3, 1, 1)
    vec_b = torch.tensor((1, 0, 0), dtype=torch.float32).view(-1, 3, 1, 1)
    gt = torch.zeros(size=(b, c, h, w), dtype=torch.float32) + vec_a
    pred = torch.zeros(size=(b, c, h, w), dtype=torch.float32) + vec_b
    # gt = torch.rand(b, c, h, w)
    # noise = torch.normal(0, std=1, size=(b, c, h, w))
    # pred = gt + noise
    
    # mask = torch.randint(0, 2, size=(b, 1, h, w))
    # from pdb import set_trace
    # set_trace()

    # loss1 = L1Accuracy_sp(pred, gt)
    # metric = MaskedMeanLpLoss(p = 1)
    # loss2 = metric(pred, gt, mask)

    # print(loss1)
    # print(loss2)
    
    # metric = MaskedSimilarityCalculator()
    # acc = metric(pred, gt, mask)
    # print(acc)
    # # print(f"{repr(metric)}: {acc.item()}")
    
    # metric = MaskedRMSECalculator()
    # acc = metric(pred, gt, mask)
    # # print(f"{repr(metric)}: {acc.item()}")
    
    # metric = MaskedAngularError()
    # acc = metric(pred, gt, mask)
    # # print(f"{repr(metric)}: {acc.item()}")