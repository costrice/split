import os

import torch
import numpy as np
from torch.overrides import is_tensor_method_or_property
import torchvision.utils
from torch import nn, optim
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm

from models.normal_autoencoder import AutoEncoder
from models.env_autoencoder import EnvAutoEncoder
from models.env_predictor import FacePredictor
from models.spherical_lighting_estimator import tensor2ndarray

from options.train_options import TrainOptions
from options.test_options import TestOptions
from losses import *

import torch.optim as optim
from utils.ibrender import IBrender
from utils.utils_lighting import *
from utils.general import *

from dataset_v2 import FaceDatasetV2
import config as config
from collections import defaultdict

from metrics import *


def test(net: nn.Module, predictor:nn.Module, test_loader, device, options):
    """
    TBA
    Args:
        net:
        test_loader:
        device:
        options:

    Returns:

    """
    
    save_dir = os.path.join(options.results_dir, f'{options.expname}_{options.num_epoch}epoch')
    if options.real_data:
        save_dir = os.path.join(options.results_dir, f'{options.expname}_{options.num_epoch}epoch_real')

    if options.saving_prefix != '':
        save_dir = save_dir + options.saving_prefix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\nUsing device {device}.\nSaving directory is {save_dir}.\n")

    ibrender = IBrender(ridx=4, envmap_size=config.sphere_scale, device=device)
    # ========================= Epoch Testing BEGIN =========================
    net.eval()
    # if net_type == 'predictor':
    #     predictor.eval()
    test_loss = defaultdict(float)
    
    # metric_lp = MaskedMeanLpLoss(p=2)
    metric_lp = MaskedRMSECalculator()
    # metric_sim = MaskedSimilarityCalculator()
    metric_light = MetricLight(device=device)

    # ========================= Epoch Testing BEGIN =========================
    
    print("testing ...")
    if options.real_data:
        num_test = len(test_loader)
    else:
        num_test = options.num_test

    name_ls = []
    name_loss_dict = {}
    
    
    fid_dir = os.path.join('cache_fid', f'{options.expname}_{options.num_epoch}_{options.saving_prefix}')
    for _dir in ['gt', 'pred']:
        os.makedirs(os.path.join(fid_dir, _dir), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # st()
            if 'env_name' in batch.keys():
                if batch['env_name'][0] not in name_loss_dict.keys():
                    env_name = batch['env_name'][0]
                    name_loss_dict[env_name] = defaultdict(float)
            env_name = batch['env_name'][0]
            name_ls.append(batch['img_name'])
            # st()
            for compo in batch:
                if type(batch[compo]) == torch.Tensor:
                    batch[compo] = batch[compo].to(device)


            ibrender.to_gpu()
            untex_hdr, tex_ldr, tex_hdr = net(batch)
            if options.use_tex:
                pred_exp_hdr = tex_hdr
                pred_exp = tex_hdr
            else:
                pred_exp_hdr = untex_hdr
                pred_exp = untex_hdr
            
            
            pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
            gt_ibr = ibrender.render_using_mirror_sphere(batch['envmap'])
            
            gt = batch['envmap']
            mask = generate_sphere_mask_from_size(batch['envmap'])
            mask = mask.to(batch['envmap'].device)
            
            # st()
            if options.output_fid:
                pred_dir = os.path.join(fid_dir, 'pred')
                gt_dir = os.path.join(fid_dir, 'gt')
                _pred_ldr = tonemapping(pred_exp * mask)
                _gt_ldr = tonemapping(gt * mask)
                # st()
                for idx, _img_name in enumerate(batch['img_name']):
                    write_image(os.path.join(pred_dir, _img_name + '.png'), 
                                tensor2ndarray(_pred_ldr[idx]))
                    write_image(os.path.join(gt_dir, _img_name + '.png'), 
                                tensor2ndarray(_gt_ldr[idx]))
                    # st()

            pred_exp_hdr = pred_exp
            batch_num = batch['envmap'].shape[0]
            
            out_metric =  metric_light(pred=pred_exp, gt=gt, mask=mask)
            for _key in out_metric:
                # if _key == 'RMSE':
                    test_loss[f'mirror_{_key}'] += out_metric[_key] * batch_num
                
            test_loss['spec_RMSE'] += metric_lp(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask) * batch_num
            test_loss['diff_RMSE'] += metric_lp(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask) * batch_num
            
            if 'env_name' in batch.keys():
                name_loss_dict[env_name]['test_num'] += 1
                name_loss_dict[env_name]['mirror_RMSE'] += metric_lp(pred=pred_exp, gt=gt, mask=mask)
                name_loss_dict[env_name]['diff_RMSE'] += metric_lp(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask)
                name_loss_dict[env_name]['spec_RMSE'] += metric_lp(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask)
            
            # Scale invaiant: normalize to mean = 0.3
            pred_mean = (torch.sum(pred_ibr['diffuse'], dim=(1,2,3), keepdim=True) / torch.sum(mask, dim=(1, 2, 3), keepdim=True)) / 3
            gt_mean = (torch.sum(gt_ibr["diffuse"], dim=(1,2,3), keepdim=True) / torch.sum(mask, dim=(1, 2, 3), keepdim=True)) / 3
        
            pred_multiplier = 0.3 / pred_mean
            pred_exp = pred_exp * pred_multiplier
            # pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
            pred_ibr['diffuse'] = pred_ibr['diffuse'] * pred_multiplier
            pred_ibr['specular'] = pred_ibr['specular'] * pred_multiplier
            
            gt_multiplier = 0.3 / gt_mean
            gt = gt * gt_multiplier
            gt_ibr['diffuse'] = gt_ibr['diffuse'] * gt_multiplier
            gt_ibr['specular'] = gt_ibr['specular'] * gt_multiplier
            
            test_loss['test_num'] += batch_num
            
            out_metric = metric_light(pred=pred_exp, gt=gt, mask=mask)
            for _key in out_metric:
                test_loss[f'si_mirror_{_key}'] += out_metric[_key] * batch_num
                
            out_metric = metric_light(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask)
            for _key in out_metric:
                test_loss[f'si_spec_{_key}'] += out_metric[_key] * batch_num
            out_metric = metric_light(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask)
            for _key in out_metric:
                test_loss[f'si_diff_{_key}'] += out_metric[_key] * batch_num
                
            if 'env_name' in batch.keys():
                name_loss_dict[env_name]['si_mirror_RMSE'] += metric_lp(pred=pred_exp, gt=gt, mask=mask)
                name_loss_dict[env_name]['si_diff_RMSE'] += metric_lp(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask)
                name_loss_dict[env_name]['si_spec_RMSE'] += metric_lp(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask)

            pred_exp = pred_exp.cpu()
            pred_ibr["diffuse"] = pred_ibr["diffuse"].cpu()
            pred_ibr["specular"] = pred_ibr["specular"].cpu()
            gt = gt.cpu()
            gt_ibr['diffuse'] = gt_ibr['diffuse'].cpu()
            gt_ibr['specular'] = gt_ibr['specular'].cpu()
            
            
            pred_exp_hdr = pred_exp
            pred_exp = pred_exp.clip(0, 1)
            for _ in pred_ibr.keys():
                pred_ibr[_] = pred_ibr[_].clip(0, 1)
                gt_ibr[_] = gt_ibr[_].clip(0, 1)

            if pred_exp.shape[-1] != 64:
                size = 64
                pred_exp_hdr = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp_hdr)
                pred_exp = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp)
                batch['envmap'] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(batch['envmap'])
                # mask = generate_sphere_mask_from_size(gt)
                for key in pred_ibr.keys():
                    pred_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_ibr[key])
                for key in gt_ibr.keys():
                    gt_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_ibr[key])
                    
            if batch_idx == 0:
                vis_num = 0
                vis_input = {}
                for key in batch.keys():
                    if type(batch[key]) == torch.Tensor:
                        vis_input[key] = [batch[key]]
                vis_pred_hdr = [pred_exp_hdr]
                vis_pred = [pred_exp]
                vis_gt_ibr = {}
                for key in gt_ibr.keys():
                    vis_gt_ibr[key] = [gt_ibr[key]]
                vis_pred_ibr = {}
                for key in pred_ibr.keys():
                    vis_pred_ibr[key] = [pred_ibr[key]]
                vis_num += pred_exp.shape[0]
                
            else:
                for k in vis_input.keys():
                    vis_input[k].append(batch[k])
                    
                vis_pred_hdr.append(pred_exp_hdr)
                vis_pred.append(pred_exp)
                for k in gt_ibr.keys():
                    vis_gt_ibr[k].append(gt_ibr[k])
                for k in pred_ibr.keys():
                    vis_pred_ibr[k].append(pred_ibr[k])
                
                vis_num += pred_exp.shape[0]

            if vis_num > num_test:
                vis_input = {k:v[:num_test] for k, v in vis_input.items()}
                vis_pred_hdr = vis_pred_hdr[:num_test]
                vis_pred = vis_pred[:num_test]
                vis_gt_ibr = {k:v[:num_test] for k, v in vis_gt_ibr.items()}
                vis_pred_ibr = {k:v[:num_test] for k, v in vis_pred_ibr.items()}
                name_ls = name_ls[:num_test]

                print(f'test {num_test} images, test end.')
                break
    
    # concate the vis results
    for k in vis_input.keys():
        vis_input[k] = torch.cat(vis_input[k], dim=0)
    vis_pred_hdr = torch.cat(vis_pred_hdr, dim=0)
    vis_pred = torch.cat(vis_pred, dim=0)
    for k in vis_gt_ibr.keys():
        vis_gt_ibr[k] = torch.cat(vis_gt_ibr[k], dim=0)
    for k in vis_pred_ibr.keys():
        vis_pred_ibr[k] = torch.cat(vis_pred_ibr[k], dim=0)

    save_loss_keys = ['si_mirror_SSIM', 'si_mirror_LPIPS', 
                      'si_mirror_RMSE',
                      'si_spec_RMSE',
                      'si_diff_RMSE',]
    _set = set(save_loss_keys)
    assert _set.issubset(test_loss.keys()), f"save_loss_keys is not a sub set of test_loss.keys()"
    # st()
    
    if 'degree_angular' in test_loss.keys():
        save_loss_keys += ['degree_angular']
        
    if len(test_loss.keys()) != 0:
        print("*" * 20)
        print(save_dir)
        print(f"Test Loss: ")
        total_loss_term = test_loss['test_num']
        for loss_term in test_loss:
            test_loss[loss_term] /= total_loss_term
            if loss_term in save_loss_keys:
                print(f"{loss_term}: {test_loss[loss_term]:>.8f} ", end="\n")
        
        print("*" * 20)
        
    full_vis, caption = draw_outputs_full(pred=vis_pred, input=vis_input, pred_render=vis_pred_ibr, gt_render=vis_gt_ibr, vis_num=num_test, vis_size=options.vis_size, face_hdr=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.utils.save_image(full_vis, os.path.join(save_dir, f'{options.expname}_{options.num_epoch}epoch_full_vis.png'))

    test_img = {}
    if 'face_face' in vis_input.keys():
        test_img['face'] = vis_input['face_face']
    elif 'face' in vis_input.keys(): 
        test_img['face'] = vis_input['face']
    test_img['GT'] = vis_input['envmap']
    test_img['GT_ldr'] = torch.clip(vis_input['envmap'], 0, 1)
    test_img['pred'] = vis_pred
    test_img['pred_hdr'] = vis_pred_hdr
    test_img['GT_ibr_diff'] = vis_gt_ibr['diffuse']
    test_img['GT_ibr_spec'] = vis_gt_ibr['specular']
    test_img['pred_ibr_diff'] = vis_pred_ibr['diffuse']
    test_img['pred_ibr_spec'] = vis_pred_ibr['specular']
    # test_img['face'] = vis_input['face']
    # st()
    save_test(save_dir=save_dir, test_img=test_img, test_loss=test_loss, options=options, name_ls=name_ls, num_test=num_test, exp_name=f'{options.expname}_{options.num_epoch}_real_{options.saving_prefix}', faltten_save=False, save_loss_keys=save_loss_keys)

    if options.real_data and not options.in_the_wild:
        save_sep_test(save_dir=save_dir, full_img=full_vis, caption=caption, name_loss_dict=name_loss_dict, options=options, num_test=num_test, exp_name=f'{options.expname}_{options.num_epoch}_real')

    print(f'Test process end.')
    del vis_input
    del vis_pred
    del vis_gt_ibr
    del vis_pred_ibr
    # ========================= Testing END =========================

if __name__ == '__main__':
    test_opts = TestOptions()
    options = test_opts.parse()
    device = options.device
    data_path = '/data4/chengyean/data/dataset_v2/test'
    
    need_ls = config.need_ls
    if options.net_type == 'autoencoder':
        need_ls = config.need_env_ls

    net_type = options.net_type
    full_set = None
    if options.real_data:
        print('real data testing.')
    if net_type == 'linet':
        full_set = FaceDatasetV2(train=False, warp2sphere=False, face_hw=64, sphere_hw=64, use_aug=options.use_aug, test_real=options.real_data, test_real_mode=options.real_test_mode, kwargs=options)
    elif net_type == 'autoencoder':
        full_set = FaceDatasetV2(train=False, warp2sphere=True, sphere_hw=64, ae=True, use_aug=options.use_aug, test_real=options.real_data, test_real_mode=options.real_test_mode, kwargs=options)
    elif net_type == 'predictor' or net_type == 'gan':
        full_set = FaceDatasetV2(train=False, warp2sphere=True, sphere_hw=64, face_hw=options.vis_size, use_aug=options.use_aug, test_real=options.real_data, test_real_mode=options.real_test_mode, kwargs=options)
    print(f"Data amount = {len(full_set)} (total test)")
    
    test_loader = DataLoader(full_set, batch_size=options.batch_size, num_workers=0)

    multi_scale = config.MULTISCALE
    net_type = options.net_type
    
    # Load Network
    exp_name = options.expname
    print(f'Experiment name: {exp_name}')
    load_dir = os.path.join(options.checkpoints_dir, exp_name)
    
    from models.spherical_lighting_estimator import SphericalLightingEstimator
    gan_est = SphericalLightingEstimator(ckpt_est=options.load_est_dir, 
                                           ckpt_pred=options.load_pred_dir, 
                                           device=device, 
                                           random_tex=options.random_tex,)
    gan_est.to(options.device)
    gan_est.mask_sp.to(options.device)
    # st()
    test(gan_est, None, test_loader, device, options)