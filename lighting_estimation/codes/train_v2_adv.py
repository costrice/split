import os

import torch
import numpy as np
from torch.overrides import is_tensor_method_or_property
import torchvision.utils
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from skimage import io
from tqdm import tqdm
import wandb

from lighting_estimation.codes.models.env_autoencoder import EnvAutoEncoder, Discriminator, MSG_Discriminator
from models.env_predictor import FacePredictor

from models.hrnet import get_hr_net

from dataset_v2 import FaceDatasetV2


from options.train_options import TrainOptions
from losses import *

import torch.optim as optim
from utils.ibrender import IBrender
from utils.utils_lighting import *

import config
from pdb import set_trace as st

import random
from collections import defaultdict

from metrics import *

import time


def train(net: nn.Module, predictor: nn.Module, train_loader, val_loader, device, options):

    # torch.autograd.set_detect_anomaly(True)
    """
    TBA
    Args:
        net: main network, linet or autoencoder.
        predictor: the predictor network. set to None for linet and autoencoder.
        train_loader:
        val_loader:
        device:
        options:

    Returns:

    """
    save_dir = os.path.join(options.checkpoints_dir, options.name)
    print(f"\nUsing device {device}.\nSaving directory is {save_dir}.\n")

    print(net)
    if predictor is not None:
        print(predictor)
 
    epochs = options.epoch
    lr = options.lr
    
    net_type = options.net_type
    if net_type == 'linet' or net_type == 'autoencoder':
        optimizer = optim.Adam(net.parameters(), lr=lr)
        if predictor is not None:
            # Discriminator
            optimizer_dis = optim.Adam(predictor.parameters(), lr=lr*0.01)
    elif net_type == 'predictor':
        optimizer = optim.Adam(predictor.parameters(), lr=lr, betas=(0.9, 0.999))
    if net_type == 'linet':
        loss_fn = loss_google(ibr=options.ibr_loss, device=device)
        # loss_fn = loss_linet(ibr=options.ibr_loss, device=device)
        if predictor!= None:
            loss_gan = RelativisticAverageHingeGAN(device, dis)
    elif net_type == 'autoencoder':
        if options.ldr_train == 1:
            print("Using LDR Training Loss")
        loss_fn = loss_autoencoder(ibr=options.ibr_loss, device=device, ldr_train=options.ldr_train, loss_ratio=config.loss_ratio)
        # loss_fn = loss_autoencoder(config.output_scale, device)
        if predictor!= None:
            loss_gan = RelativisticAverageHingeGAN(device, dis)

    elif net_type == 'predictor':
        loss_fn = loss_predictor(ibr=options.ibr_loss, device=device)

    if net_type == 'autoencoder':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.ae_scheduler_step, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.scheduler_step, gamma=0.1)
    # initialize wandb logging

    print("Project Name: ", map_expname(net_type))
    wandb.init(project=map_expname(net_type), entity="cilab-lighting",
                name=options.name,
                config={
                   "learning_rate": lr,
                   "lr_schedule": "None",
                   "batch_size": options.batch_size,
                   "envmap_size": config.sphere_scale,
                   "face_input_size": config.input_face_scale,
                   "latent_size": config.latent_scale,
                   "use_aug": options.use_aug,
               })
    if net_type == 'predictor':
        wandb.config.update({'ae_dir': load_dir})
    if net_type == 'linet':
        wandb.config.update({'linet_input_size': config.input_face_scale})
    wandb.watch(net)

    if options.ibr_loss:
        print("Using IBR loss.")
    else:
        print("Do not use IBR loss.")
    iter = 0
    log_step = 0
    log_meter = 0
    running_loss = defaultdict(float)

    ibrender = IBrender(ridx=4, envmap_size=config.sphere_scale, device=device, ibr_setting=options.ibr_loss)
    
    metric_lp = MaskedMeanLpLoss(p=1)
    metric_sim = MaskedSimilarityCalculator()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n----------------------------------")
        # ========================= Epoch Training BEGIN =========================
        if net_type == 'linet' or net_type == 'autoencoder':
            net.train()
        elif net_type == 'predictor':
            net.eval()
            for param in net.parameters():
                param.requires_grad = False
            predictor.train()

        ibrender.to_gpu()
        if epoch == options.gan_early_epoch:
            print('Start GAN Training! ')

        train_vis_flag = True
        # torch.autograd.set_detect_anomaly(True)
        # start_time = time.time()
        time_dir = {}
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            for compo in batch:
                if compo != 'name':
                    batch[compo] = batch[compo].to(device)
            # time_dir['data_time'] = time.time() - start_time
            if net_type == 'linet':
                # st()
                if options.use_gan == 0:
                    pred, pred_ds2, pred_ds4, pred_ds8 = net(batch)
                    # Multiscale pred and gt
                    pred_dict = {'fs': pred, 'ds2': pred_ds2, 'ds4': pred_ds4, 'ds8': pred_ds8}
                    gt_dict = generate_real_dict_google(batch['envmap'])
                    # st()
                    # render with ibrender
                    pred_render_dict = {}
                    for key in pred_dict.keys():
                        size = pred_dict[key].shape[-1]
                        pred_dict[key] = torch.clamp(pred_dict[key], 0, 15)
                        pred_exp_scale = torch.exp(pred_dict[key]) - 1
                        pred_exp_scale = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp_scale)
                        pred_render_scale = ibrender.render_using_mirror_sphere(pred_exp_scale)
                        pred_render_scale['mirror'] = pred_exp_scale
                        for render in pred_render_scale.keys():
                            pred_render_scale[render] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_render_scale[render])
                        pred_render_dict[key] = pred_render_scale

                    gt_render_dict = {}
                    for key in gt_dict.keys():
                        size = gt_dict[key].shape[-1]
                        gt_exp_scale = gt_dict[key]
                        gt_exp_scale = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_exp_scale)
                        gt_render_scale = ibrender.render_using_mirror_sphere(gt_exp_scale)
                        gt_render_scale['mirror'] = gt_exp_scale
                        for render in gt_render_scale.keys():
                            gt_render_scale[render] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_render_scale[render])
                        gt_render_dict[key] = gt_render_scale
                    # st()

                    # verbose 
                    # for key1 in pred_render_dict:
                    #     for key2 in pred_render_dict[key1]:
                    #         print(f"pred_{key1}_{key2}: {pred_render_dict[key1][key2].max(), pred_render_dict[key1][key2].min()}")
                    #         # print(f"{key1}_{key2}: {gt_render_dict[key1][key2].max(), gt_render_dict[key1][key2].min()}")
                    # for key1 in pred_render_dict:
                    #     for key2 in pred_render_dict[key1]:
                    #         print(f"gt_{key1}_{key2}: {gt_render_dict[key1][key2].max(), gt_render_dict[key1][key2].min()}")
                    #         # print(f"{key1}_{key2}: {gt_render_dict[key1][key2].max(), gt_render_dict[key1][key2].min()}")


                    # st()
                    loss = loss_fn(pred_render_dict, gt_render_dict)
                else:
                    pred, pred_ds2, pred_ds4, pred_ds8 = net(batch)
                    # Multiscale pred and gt
                    pred_dict = {'fs': pred, 'ds2': pred_ds2, 'ds4': pred_ds4, 'ds8': pred_ds8}
                    gt_dict = generate_real_dict_google(batch['envmap'])
                    # st()
                    # render with ibrender
                    pred_render_dict = {}
                    for key in pred_dict.keys():
                        size = pred_dict[key].shape[-1]
                        pred_dict[key] = torch.clamp(pred_dict[key], 0, 15)
                        pred_exp_scale = torch.exp(pred_dict[key]) - 1
                        pred_exp_scale = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp_scale)
                        pred_render_scale = ibrender.render_using_mirror_sphere(pred_exp_scale)
                        pred_render_scale['mirror'] = pred_exp_scale
                        for render in pred_render_scale.keys():
                            pred_render_scale[render] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_render_scale[render])
                        pred_render_dict[key] = pred_render_scale

                    gt_render_dict = {}
                    for key in gt_dict.keys():
                        size = gt_dict[key].shape[-1]
                        gt_exp_scale = gt_dict[key]
                        gt_exp_scale = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_exp_scale)
                        gt_render_scale = ibrender.render_using_mirror_sphere(gt_exp_scale)
                        gt_render_scale['mirror'] = gt_exp_scale
                        for render in gt_render_scale.keys():
                            gt_render_scale[render] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_render_scale[render])
                        gt_render_dict[key] = gt_render_scale
                    loss = loss_fn(pred_render_dict, gt_render_dict)
                    if epoch > options.gan_early_epoch:
                        # clip to mirror representation and normalize to [-1, 1]
                        pred_gan_dict = {}
                        for key in pred_render_dict.keys():
                            pred_render_dict[key]['mirror'] = torch.clip(pred_render_dict[key]['mirror'], 0, 1)
                            pred_gan_dict[key] = normalize(pred_render_dict[key]['mirror'])
                        gt_gan_dict = {}
                        for key in gt_render_dict.keys():
                            gt_render_dict[key]['mirror'] = torch.clip(gt_render_dict[key]['mirror'], 0, 1)
                            gt_gan_dict[key] = normalize(gt_render_dict[key]['mirror'])
                        # st()
                        # loss['gan'] = 0.01 * loss_gan.gen_loss(gt_dict, pred_dict)
                        loss['gan'] = 0.001 * loss_gan.gen_loss(gt_gan_dict, pred_gan_dict)
                        loss['total'] += loss['gan']

            elif net_type == 'autoencoder':
                
                # st()
                if options.use_gan == 0:
                    pred, _, _ = net(batch['envmap'])
                    pred_exp = torch.exp(pred) - 1
                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(batch['envmap'])
                    loss = loss_fn(pred, batch['envmap'], pred_ibr, gt_ibr, epoch)
                    # if epoch > 5:
                    #     loss['gan'] = 0.01 * loss_gan.gen_loss(batch['envmap'], pred_exp)
                    #     loss['total'] += loss['gan']
                else:
                    pred, pred_ds2, pred_ds4 = net(batch['envmap'])
                    pred_exp = torch.exp(pred) - 1
                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(batch['envmap'])
                    loss = loss_fn(pred, batch['envmap'], pred_ibr, gt_ibr, epoch)
                    if epoch > options.gan_early_epoch:
                        pred_dict = {'fs': pred, 'ds2': pred_ds2, 'ds4': pred_ds4}
                        gt_dict = generate_real_dict(batch['envmap'])
                        # clip to mirror representation and normalize to [-1, 1]
                        for key in pred_dict:
                            pred_dict[key] = torch.clip(pred_dict[key], 0, 1)
                            pred_dict[key] = normalize(pred_dict[key])
                        for key in gt_dict:
                            gt_dict[key] = torch.clip(gt_dict[key], 0, 1)
                            gt_dict[key] = normalize(gt_dict[key])
                        # st()
                        # loss['gan'] = 0.01 * loss_gan.gen_loss(gt_dict, pred_dict)
                        loss['gan'] = 1 * loss_gan.gen_loss(gt_dict, pred_dict)
                        loss['total'] += loss['gan']

            elif net_type == 'predictor':
                # st()
                rec_code = net.encoder_forward(batch['envmap']).detach()
                pred_code = predictor(batch)
                pred = net.decoder_forward(pred_code)
                loss = loss_fn(pred_code, pred, batch['envmap'], rec_code)
                pred_exp = torch.exp(pred) - 1

            # time_dir['net_time'] = time.time() - start_time - time_dir['data_time']
            if options.use_gan == 1 and (net_type == 'autoencoder' or net_type == 'linet'):
                if epoch > options.gan_early_epoch:
                    if batch_idx % 4 == 0:
                        # print('Train Discriminator! ')
                        for p in pred_dict:
                            pred_dict[p] = pred_dict[p].detach()
                        # loss_gan_dis = 0.001 * loss_gan.dis_loss(gt_dict, pred_dict)
                        if net_type == 'autoencoder':
                            loss_gan_dis = 0.1 * loss_gan.dis_loss(gt_dict, pred_dict)
                        else:
                            loss_gan_dis = 0.1 * loss_gan.dis_loss(gt_gan_dict, pred_gan_dict)

                        running_loss['dis_loss'] = loss_gan_dis.item()
                        optimizer_dis.zero_grad()
                        loss_gan_dis.backward()
                        optimizer_dis.step()
                    else:
                        # print('Train Generator! ')
                        optimizer.zero_grad()
                        loss["total"].backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    loss["total"].backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     loss["total"].backward()
                loss["total"].backward()
                optimizer.step()

            # time_dir['optim_time'] = time.time() - start_time - time_dir['net_time']
            for loss_term in loss:
                running_loss[loss_term] += loss[loss_term].item()



            iter += 1
            log_meter += 1
            # print(f"Time: {time_dir['data_time']:.2f}s, {time_dir['net_time']:.2f}s, {time_dir['optim_time']:.2f}s")
            
            if iter % options.log_batch_freq == 0 and train_vis_flag:
                if type(pred) == tuple:
                    pred = pred[0]
                pred_exp = torch.exp(pred) - 1
                vis, cap = draw_outputs_full(pred=pred_exp, input=batch)
                pred_visualization = wandb.Image(vis, mode="RGB", caption=cap)

                log_dict = {"prediction": pred_visualization}

                for loss_term in running_loss:
                    log_dict["loss_" + loss_term] = running_loss[loss_term] / log_meter
                # log_dict['total_vgg'] = (running_loss['total'] + running_loss['vgg']) / log_meter
                for loss_term in running_loss:
                    running_loss[loss_term] = 0
                wandb.log(log_dict, step = log_step)
                log_meter = 0
                train_vis_flag = False
                
        # ========================= Epoch Training END =========================
        
        # ========================= Epoch Testing BEGIN =========================
        print("Start Testing...")
        net.eval()
        if net_type == 'predictor':
            predictor.eval()
        test_loss = defaultdict(float)

        vis_cnt = 0

        # max_dict = defaultdict(float)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                for compo in batch:
                    if compo != 'name':
                        batch[compo] = batch[compo].to(device)
                
                ibrender.to_gpu()
                if net_type == 'autoencoder':
                    pred = net(batch['envmap'])
                    if type(pred) == tuple:
                        pred = pred[0]
                    pred_exp = torch.exp(pred) - 1
                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(batch['envmap'])
                    loss = loss_fn(pred, batch['envmap'], pred_ibr, gt_ibr, epoch)
                    
                elif net_type == 'predictor':
                    rec_code = net.encoder_forward(batch['envmap']).detach()
                    pred_code = predictor(batch)
                    pred = net.decoder_forward(pred_code)
                    pred_exp = torch.exp(pred) - 1
                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(batch['envmap'])
                    loss = loss_fn(pred_code, pred, batch['envmap'], rec_code)

                for loss_term in loss:
                    test_loss[loss_term] += loss[loss_term].item()

                #### METRIC CALCULATION

                # preprocess
                ibrender.to_cpu()
                pred = detach_and_cpu(pred)

                if pred.shape[-1] != 64:
                    size = pred.shape[-1]
                    pred = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred)
                    batch['envmap'] = torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(batch['envmap'])

                    pred_exp = torch.exp(pred) - 1
                    gt = detach_and_cpu(batch['envmap'])
                    
                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(gt)

                    pred_exp = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp)
                    gt = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt)
                    mask = generate_sphere_mask_from_size(gt)
                    for key in pred_ibr.keys():
                        pred_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_ibr[key])
                    for key in gt_ibr.keys():
                        gt_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_ibr[key])

                else:
                    pred_exp = torch.exp(pred) - 1
                    gt = detach_and_cpu(batch['envmap'])
                    mask = generate_sphere_mask_from_size(batch['envmap'])

                    pred_ibr = ibrender.render_using_mirror_sphere(pred_exp)
                    gt_ibr = ibrender.render_using_mirror_sphere(gt)

                test_loss['envmap_l1_accuracy'] += metric_lp(pred=pred_exp, gt=gt, mask=mask)
                test_loss['diff_l1_accuracy'] += metric_lp(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask)
                test_loss['spec_l1_accuracy'] += metric_lp(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask)

                # calculate metric for ldr probe
                pred_exp = torch.clip(pred_exp, 0, 1)
                gt = torch.clip(gt, 0, 1)
                pred_ibr = {k: torch.clip(v, 0, 1) for k, v in pred_ibr.items()}
                gt_ibr = {k: torch.clip(v, 0, 1) for k, v in gt_ibr.items()}

                sim_res = metric_sim(pred_exp, gt, mask)
                for key in sim_res.keys():
                    test_loss['ldr_envmap_similarity_' + key] += sim_res[key]

                test_loss['ldr_mirror_l1_accuracy'] += metric_lp(pred=pred_exp, gt=gt, mask=mask)
                test_loss['ldr_diff_l1_accuracy'] += metric_lp(pred=pred_ibr["diffuse"], gt=gt_ibr["diffuse"], mask=mask)
                test_loss['ldr_spec_l1_accuracy'] += metric_lp(pred=pred_ibr["specular"], gt=gt_ibr["specular"], mask=mask)

                if pred_exp.shape[-1] != 64:
                    size = 64
                    pred_exp = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_exp)
                    batch['envmap'] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(batch['envmap'])
                    # mask = generate_sphere_mask_from_size(gt)
                    for key in pred_ibr.keys():
                        pred_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(pred_ibr[key])
                    for key in gt_ibr.keys():
                        gt_ibr[key] = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(gt_ibr[key])

                if batch_idx == 0:
                    vis_input = batch
                    if 'name' in vis_input.keys():
                        vis_input.pop('name')
                    vis_pred = pred_exp
                    vis_gt_ibr = gt_ibr
                    vis_pred_ibr = pred_ibr
                    vis_cnt += pred.shape[0]
                else:
                    if vis_cnt <= 50:
                        for k in batch.keys():
                            if k != 'name':
                                vis_input[k] = torch.cat([vis_input[k], batch[k]], dim=0)
                            
                        vis_pred = torch.cat([vis_pred, pred_exp], dim=0)
                        for k in gt_ibr.keys():
                            vis_gt_ibr[k] = torch.cat([vis_gt_ibr[k], gt_ibr[k]], dim=0)
                        for k in pred_ibr.keys():
                            vis_pred_ibr[k] = torch.cat([vis_pred_ibr[k], pred_ibr[k]], dim=0)
                        vis_cnt += pred_exp.shape[0]
                        
        print(f"Test Loss: ", end="")
        for loss_term in test_loss:
            test_loss[loss_term] /= len(val_loader)
            print(f"{loss_term}: {test_loss[loss_term]:>.8f} ", end="")
        print("")
        test_log_dict = {"test loss " + i: test_loss[i] for i in test_loss.keys()}
        
        # visualization
        temp_ls = [i for i in range(vis_pred.shape[0])]
        # rand_idx = random.sample(temp_ls, 12)
        rand_idx = temp_ls[:12]

        vis_input = {k: v[rand_idx] for k, v in vis_input.items()}
        vis_pred = vis_pred[rand_idx]
        vis_gt_ibr = {k: v[rand_idx] for k, v in vis_gt_ibr.items()}
        vis_pred_ibr = {k: v[rand_idx] for k, v in vis_pred_ibr.items()}

        vis, cap = draw_outputs_full(pred=vis_pred, input=vis_input, gt_render=vis_gt_ibr, pred_render=vis_pred_ibr)
        pred_visualization = wandb.Image(vis, mode="RGB", caption=cap)
        
        del vis_input
        del vis_pred
        del vis_gt_ibr
        del vis_pred_ibr

        test_log_dict["test prediction"] = pred_visualization
                
        wandb.log(test_log_dict, step=log_step)
        # ========================= Testing END =========================
         
        if (epoch + 1) % options.save_epoch_freq == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if net_type == 'linet' or net_type == 'autoencoder':
                torch.save(net.state_dict(), os.path.join(save_dir, f"epoch{epoch + 1}.pth"))
            elif net_type == 'predictor':
                torch.save(predictor.state_dict(), os.path.join(save_dir, f"epoch{epoch + 1}.pth"))

        scheduler.step()
        log_step += 1

        #debug
        # torchvision.utils.save_image(vis, "1213_debug.png")
        # break
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, f"final.pth"))
    


if __name__ == '__main__':
    train_opts = TrainOptions()
    options = train_opts.parse()
    # set device
    device = options.device
    # linet: pipeline 0
    # autoencoder: autoencoder in pipeine 1
    # predictor: predictor in pipeline 1
    net_type = options.net_type

    need_ls = config.need_ls
    if options.net_type == 'autoencoder':
        need_ls = config.need_env_ls

    full_set = None
    print("Using dataset: ", options.env_datasets)

    finetune = False
    if options.load_pred_dir != '':
        print('Fine-tuning from', options.load_pred_dir)
        finetune = True
    

    if net_type == 'linet':
        full_set = FaceDatasetV2(train=True, warp2sphere=False, face_hw=64, sphere_hw=64, use_aug=options.use_aug, env_datasets=options.env_datasets)
    elif net_type == 'autoencoder':
        full_set = FaceDatasetV2(train=True, warp2sphere=True, sphere_hw=64, ae=True, use_aug=options.use_aug, env_datasets=options.env_datasets)
    elif net_type == 'predictor':
        full_set = FaceDatasetV2(train=True, warp2sphere=True, sphere_hw=64, use_aug=options.use_aug, env_datasets=options.env_datasets, read_from_sphere=options.read_from_sphere, finetune=finetune)

    val_size = int(len(full_set) * options.val_rate)
    train_size = len(full_set) - val_size
    train_set, val_set = random_split(full_set, [train_size, val_size])
    print(f"Data amount = {len(full_set)} (total), {len(train_set)} (train), {len(val_set)} (valid)\n")
    
    train_loader = DataLoader(train_set, batch_size=options.batch_size, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=options.batch_size, num_workers=16, shuffle=True)
    # create network
    multi_scale = config.MULTISCALE
    
    gan = options.use_gan
    if net_type == 'autoencoder':
        # 0127 2048 training
        # net = EnvAutoEncoder(vector_size=2048, sphere_size=config.sphere_scale)
        net = EnvAutoEncoder(vector_size=config.latent_scale, sphere_size=config.sphere_scale)
        net.to(options.device)
        dis = None
        if gan:
            dis = MSG_Discriminator()
            dis.to(options.device)
        train(net, dis, train_loader, val_loader, device, options)
        
    elif net_type == 'predictor':
        net_autoencoder = EnvAutoEncoder(vector_size=config.latent_scale, sphere_size=config.sphere_scale)
        load_dir = options.load_ae_dir
        model_dict = net_autoencoder.load_state_dict(torch.load(load_dir, map_location=torch.device('cpu')))
        net_autoencoder.to(options.device)
        if options.predictor_type == 'cyanet':
            net_predictor = FacePredictor(train_setting=config.training_need_ls_sp, img_hw=config.sphere_scale, code_scale=config.latent_scale)
        elif options.predictor_type == 'hrnet':
            if options.compo_ab == 0:
                net_predictor = get_hr_net(config.training_need_ls_sp)
            elif options.compo_ab == 1:
                train_setting = config.training_need_ls_sp
                train_setting.remove('shading')
                net_predictor = get_hr_net(train_setting)
            elif options.compo_ab == 2:
                train_setting = config.training_need_ls_sp
                train_setting.remove('specular')
                net_predictor = get_hr_net(train_setting)
            elif options.compo_ab == 3:
                train_setting = config.training_need_ls_sp
                train_setting.remove('normal')
                net_predictor = get_hr_net(train_setting)
            
            if options.load_pred_dir != '':
                load_pred_dir = options.load_pred_dir
                net_predictor.load_state_dict(torch.load(load_pred_dir, map_location=torch.device('cpu')))
                print(f"Finish Loading from: {load_pred_dir}")
            
        net_predictor.to(options.device)
        train(net_autoencoder, net_predictor, train_loader, val_loader, device, options)
