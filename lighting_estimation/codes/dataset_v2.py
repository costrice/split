# -*- coding: utf-8 -*-

import os
import random
import time
from functools import partial
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from utils.utils_lighting import generate_sphere_mask_from_size_np
import config 
from utils.general import \
    linrgb2srgb, \
    read_image, \
    write_image, \
    read_mask

# my own utils and configs
from utils.render import collect_rendered
from utils.warp import generate_sphere_mask_and_normal, warp_face_to_sphere, warp_sp2rect

from tqdm import tqdm

from copy import deepcopy

from pdb import set_trace as st

import time

def read_info(txt_file: str):
    """Read data group information (random params used when rendering) from the
    txt file, whose path is `txt_file`.
    
    Args:
        txt_file: path to the txt file.

    Returns:
        Dict[str, ?]: Parsed information.
    """
    info = {}
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split(" ", 1)
            value = value.rstrip()
            if key in ["focal_length", "face_distance", "envmap_spin",
                       "specular_intensity", "specular_roughness", "displacement_scale"]:
                value = float(value)
            elif key in ["face_rotation"]:  # list
                value = [float(number) for number in value[1: -2].split()]
            info[key] = value
    return info
        

def show_batch(img_batch: torch.Tensor,
               save_folder: str,
               nrow: int,
               title: str):  # figsize=(40, 20),):
    """
    Show several torch Tensor in grids.
    """
    save_folder = os.path.join(config.dir_examples, save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    img_batch = img_batch.detach().cpu()
    grid = torchvision.utils.make_grid(img_batch, nrow=nrow)
    grid = grid.numpy().transpose((1, 2, 0))
    
    write_image(os.path.join(save_folder, title + ".png"), grid)


def random_resized_crop(sample: Dict[str, torch.tensor],
                        max_mag: float):
    """
    Performs an identical random crop on all face-shaped components in `sample`,
    then resizes to the original size. Should only be used in face component.
    
    Args:
        sample: the data group to be processed
        max_mag: maximum magnitude of augmentation.
        
    Returns:
        Dict[str, np.ndarray]: processed data group.
    """
    orig_hw = sample["mask"].shape[-1]
    # st()
    after_hw = np.random.randint(int(orig_hw * (1 - max_mag)), orig_hw)
    # random crop
    top = np.random.randint(0, orig_hw - after_hw + 1)
    left = np.random.randint(0, orig_hw - after_hw + 1)
    right = left + after_hw
    bottom = top + after_hw
    
    # ensure that there are some face region appeared in the image.
    # while np.sum(sample["mask"][top: bottom, left: right, :]) == 0:
    #     top = np.random.randint(0, orig_hw - after_hw + 1)
    #     left = np.random.randint(0, orig_hw - after_hw + 1)
    #     right = left + after_hw
    #     bottom = top + after_hw
    while torch.sum(sample["mask"][:, top: bottom, left: right]) == 0:
        top = np.random.randint(0, orig_hw - after_hw + 1)
        left = np.random.randint(0, orig_hw - after_hw + 1)
        right = left + after_hw
        bottom = top + after_hw

    for component, img in sample.items():
        if component in ["normal", "albedo", "face", "shading", "specular", "mask"]:
            # interp = cv2.INTER_NEAREST if component in ["mask"] else cv2.INTER_LINEAR
            # c = img.shape[-1]
            # img = cv2.resize(img[top: bottom, left: right, :], dsize=(orig_hw, orig_hw), interpolation=interp)
            img = transforms.Resize((orig_hw, orig_hw))(img[:, top: bottom, left: right])
            # if c == 1:  # cv2.resize will discard this dimension
            #     img = np.resize(img, (*img.shape, 1))
        elif component in ["envmap"]:  # don't crop or resize
            pass
        sample[component] = img
    return sample


def random_horizontal_flip(sample: Dict[str, torch.tensor],
                           max_mag: float = 0):
    """Performs an identical random horizontal flip on all components in
    `sample` and modifies "normal" accordingly. """
    if np.random.random() > 0.5:
        for component, img in sample.items():
            # img = img[:, ::-1, :].copy()
            # img = img[:, :, ::-1].copy()
            img = transforms.RandomHorizontalFlip(p=1)(img)
            if component == "normal":  # flip normal direction
                # img[:, :, 0] = -img[:, :, 0]
                img[0, :, :] = -img[0, :, :]
            sample[component] = img
    return sample


def random_exposure(sample: Dict[str, torch.tensor],
                    max_mag: float):

    """Applies exposure distortion to light-relative components. """
    exposure = 2 ** (np.random.normal(loc=0, scale=max_mag / 2.5))
    for component, img in sample.items():
        if component in ["face", "shading", "specular", "envmap"]:
            img = img * exposure
        sample[component] = img
    return sample


def random_white_balance(sample: Dict[str, torch.tensor],
                         max_mag: float):
                         
    """Applies white balance distortion to light-relative components. """
    # white_balance = np.random.uniform(1 - max_mag, 1 + max_mag, size=(1, 1, 3)).astype(np.float32)
    white_balance = np.random.uniform(1 - max_mag, 1 + max_mag, size=(3, 1, 1)).astype(np.float32)
    for component, img in sample.items():
        if component in ["face", "shading", "specular", "envmap"]:
            img = img * white_balance
        sample[component] = img
    return sample


def random_noise_sphere(sample: Dict[str, torch.tensor],
                        max_mag: float):
    """Applies multiplicative random noise to "shading" and "specular"
    component (used as input to net). """
    if not ('shading' in sample and 'specular' in sample):
        return sample
    for component in ["shading", "specular"]:
        img = sample[component]
        mag = np.random.uniform(0, max_mag)
        noise = np.random.normal(loc=0, scale=mag / 2.5,
                                 size=(img.shape[0], img.shape[1], 1))
        noise = torch.from_numpy(noise).float()
        img = img * (2 ** noise)
        sample[component] = img
    return sample


def random_noise_face(sample: Dict[str, torch.tensor],
                      max_mag: float):
    """Applies multiplicative random noise to "face" component
    (used as input to net). """
    if "face" not in sample:
        return sample
    for component in ["face"]:
        img = sample[component]
        mag = np.random.uniform(0, max_mag)
        noise = np.random.normal(loc=0, scale=mag / 2.5, size=img.shape)
        noise = torch.from_numpy(noise).float()
        img = img * (2 ** noise)
        sample[component] = img
    return sample
    

def identity(sample: Dict[str, torch.tensor],
             max_mag: float):
    """Returns the input `sample` as it is."""
    return sample


class RandAugment:
    """Applies random augmentation to a data group."""
    def __init__(self,
                 n: int,
                 m: int,
                 augment_list: List[str] = None):
        self.n = n  # number of augmentations applied each time
        self.m = m  # augmentation magnitude
        
        # add augmentations
        if augment_list is None:
            augment_list = ["crop", "exposure", "white_balance"]
        self.augment_list = [(identity, 0, 0)]
        for augment in augment_list:
            if augment == "crop":
                self.augment_list.append((random_resized_crop, 0, 0.5))
            elif augment == "exposure":
                self.augment_list.append((random_exposure, 0, 3))
                #cya: modified to a smaller scale
                # self.augment_list.append((random_exposure, 0, 1))
            elif augment == "white_balance":
                self.augment_list.append((random_white_balance, 0, 0.3))
            elif augment == "noise_sphere":
                self.augment_list.append((random_noise_sphere, 0, 0.15))
            elif augment == "noise_face":
                self.augment_list.append((random_noise_face, 0, 0.10))
    
    def __call__(self, sample, show_res=False):
        # do random flip anyway
        sample = random_horizontal_flip(sample)
        # do other random augmentation
        ops = random.choices(self.augment_list, k=self.n)
        mag = float(self.m) / 30
        for op, min_mag, max_mag in ops:
            sample = op(sample, (max_mag - min_mag) * mag + min_mag)
        return sample


class FaceDatasetV2(Dataset):
    """A torch.Dataset object for rendered face images,
   based on FaceScape 3D dataset and Laval HDR Environment Map datasets.
   
   Dataset References:
       FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction,
       Deep Sky Modeling for Single Image Outdoor Lighting Estimation,
       Learning to Predict Indoor Illumination from a Single Image.
   """
    def __init__(self,
                 train: bool,
                 warp2sphere: bool,
                 face_hw: int = 512,
                 sphere_hw: int = 64,
                 dataset_dir: str = None,
                 env_datasets: List[str] = None,
                 ae: bool = False,
                 use_aug: bool = True,
                 test_real: bool = False,
                 read_from_sphere: bool = False,
                 test_real_mode: int = 0,
                 google_net: bool = False,
                 finetune: bool = False,
                 rect_envmap: bool = False,
                 **kwargs
                 ):
        """Prepare for reading images.
        
        Args:
            train:
                Whether regards this dataset as training set or testing set.
                Affects data path and data augmentation.
            warp2sphere:
                If True, return warped "shading", "specular", "mask",
                "normal" (all above are sphere) and "info".
                if False, return components in `config.compo_filename`.
            face_hw: only used when `warp2sphere`==False.
                Face image will be resized to (`face_hw`, `face_hw`), also used in real test, when `warp2sphere`==True.
            sphere_hw: only used when `warp2sphere`==True.
                Sphere image will be generated as `sphere_hw`, `sphere_hw`.
            dataset_dir: Should be used when dataset to be read is
                not as indicated in `config.dir_rendered_dataset`.
            env_datasets: Should be `None` (which mean include all)
                or a subset of `["indoor", "outdoor", "sky"]`.
                Indicates which part of rendered dataset should be included
                according to the envmap dataset used to render them.
            ae: whether use autoencoder training, if used, just read envmap
                for quick training
            test_real: whether use real test data, if used, modify the data path preperation
            read_from_sphere: only used in sphere training, if true, just read from dataset_v2_sphere
            test_real_mode: only used in real test, if 0, test indoor + outdoor, if 1, test indoor only, if 2, test outdoor only if 3 do not read from excel but read from walking folder config.visual_data_dir
            google_net: linet version. if True, use google net, if False, use tdv net
            fintune: whether to use finetune dataset. only used in proposed predictor training.
            rect_envmap: whether to use rectified envmap. only used in 3dv training.
        """
        super(FaceDatasetV2, self).__init__()
        
        
        phase = "train" if train else "test"
        self.train = train
        self.ae = ae
        self.read_from_sphere = read_from_sphere
        if self.read_from_sphere:
            print("read from sphere")
        
        try:
            self.gt_test = kwargs['kwargs'].gt_test
        except:
            self.gt_test = False
        if self.gt_test:
            print("gt test")
            
        try: 
            self.in_the_wild = kwargs['kwargs'].in_the_wild
        except:
            self.in_the_wild = False
        if self.in_the_wild:
            print("in the wild")
            
        try: 
            self.new_outdoor = kwargs['kwargs'].new_outdoor
        except:
            self.new_outdoor = False
        if self.new_outdoor:
            print("new_outdoor")
            
        try: 
            self.syn_real_v2 = kwargs['kwargs'].syn_real_v2
        except:
            self.syn_real_v2 = False
        if self.syn_real_v2:
            print("syn_real_v2")
            
        try: 
            self.syn_real = kwargs['kwargs'].syn_real
        except:
            self.syn_real = False
        if self.syn_real:
            print("syn_real")
        # st()
        assert not (self.syn_real and self.syn_real_v2)
        
        try: 
            self.rerender_loss = kwargs['kwargs'].rerender_loss
            self.setup_rerender_config()
        except:
            self.rerender_loss = False
        
        # collect rendered data groups
        if not self.read_from_sphere:
            if dataset_dir is None:
                dataset_dir = config.dir_rendered_dataset[phase]
            self.compo_filename = config.compo_filename
        else:
            self.compo_filename = config.compo_filename_sphere
            if dataset_dir is None:
                if self.train:
                    dataset_dir = config.dir_rendered_dataset_sphere[phase]
                else:
                    dataset_dir = config.dir_rendered_dataset_finetune[phase]
                    self.compo_filename = config.compo_filename_sphere_finetune
                    # st()

        # if not self.read_from_sphere:
        #     self.compo_filename = config.compo_filename
        # else:
        #     self.compo_filename = config.compo_filename_sphere
        self.finetune = finetune
        if self.finetune or self.train == False:
            if warp2sphere:
                dataset_dir = config.dir_rendered_dataset_finetune[phase]
                self.compo_filename = config.compo_filename_sphere_finetune
            else:
                dataset_dir = config.dir_rendered_dataset_finetune_face[phase]
                self.compo_filename = config.compo_filename_face_finetune
                if not self.gt_test:
                    self.compo_filename.pop('face')
        
        print("dataset_dir:", dataset_dir)
        print("compo_filename:", self.compo_filename)
        # st()
            
        if env_datasets is None:
            env_datasets = ["indoor", "outdoor", "sky"]
        self.dataset_dir = dataset_dir
        # st()
        try: 
            data_group_dict = collect_rendered(train=train, load=True, dataset_dir=dataset_dir, compo_dict=self.compo_filename)
            # st()
            self.data_group_dirs = []
            # only leave data rendered with selected envmap datasets
            for dataset in env_datasets:
                self.data_group_dirs += data_group_dict[dataset]
        except:
            print("fail to load data from " + dataset_dir)
        self.rec_envmap = rect_envmap
        
        self.test_real = test_real
        self.test_real_mode = test_real_mode
        if self.train == False:
            if self.test_real:
                
                if not self.new_outdoor:                                                                                
                    if self.syn_real:
                        csv_dir = config.csv_dir_realtest_old_syn_real 
                    elif self.syn_real_v2:
                        csv_dir = config.csv_dir_realtest_old_syn_real_v2
                    else:
                        csv_dir = config.csv_dir_realtest_old_syn 
                else:
                    if self.syn_real:
                        csv_dir = config.csv_dir_laval_syn_real 
                    elif self.syn_real_v2:
                        csv_dir = config.csv_dir_laval_syn_real_v2
                    else:
                        csv_dir = config.csv_dir_laval_syn 
                # st()
                print("csv_dir:", csv_dir)
                if self.in_the_wild:
                    if self.syn_real:
                        csv_itw_dir = config.csv_dir_FFHQ_inthewild_syn_real
                    elif self.syn_real_v2:
                        csv_itw_dir = config.csv_dir_FFHQ_inthewild_syn_real_v2
                    else:
                        csv_itw_dir = config.csv_dir_FFHQ_inthewild_syn
                
                    print("csv_dir:", csv_itw_dir)
                import pandas as pd
                if test_real_mode == 0:
                    if self.in_the_wild:
                        df = pd.read_csv(csv_itw_dir)
                        self.data_group_dirs = df['0'].tolist()
                    else:
                        df = pd.read_csv(csv_dir)
                        self.data_group_dirs = df['0'].tolist()
                        self.data_group_dirs = self.data_group_dirs
                elif test_real_mode == 1:
                    if self.in_the_wild:
                        df = pd.read_csv(csv_itw_dir)
                        self.data_group_dirs = df['0'].tolist()
                    else:
                        df = pd.read_csv(csv_dir)
                        self.data_group_dirs = df['0'].tolist()
                        self.data_group_dirs = [i for i in self.data_group_dirs if 'indoor' in i]
                elif test_real_mode == 2:
                    if self.in_the_wild:
                        df = pd.read_csv(csv_itw_dir)
                        self.data_group_dirs = df['0'].tolist()
                    else:
                        df = pd.read_csv(csv_dir)
                        self.data_group_dirs = df['0'].tolist()
                        self.data_group_dirs = [i for i in self.data_group_dirs if 'outdoor' in i or 'LavalFaceLighting' in i]
                self.compo_filename = config.compo_real_data
                self.compo_filename_laval = config.compo_real_data_laval

            else:
                data_group=None
                if test_real_mode == 0:
                    data_group = ["indoor", "outdoor", "sky"]
                elif test_real_mode == 1:
                    data_group = ["indoor"]
                elif test_real_mode == 2:
                    data_group = ["outdoor", "sky"]
                # st()
                if self.read_from_sphere:
                    import pandas as pd
                    data_csv = None
                    if test_real_mode == 0:
                        raise NotImplementedError
                    elif test_real_mode == 1:
                        # df = pd.read_csv('/data4/chengyean/data/indoor_syn_test.csv')
                        if self.gt_test:
                            data_csv = config.csv_dir_syntest_gt_sp_indoor
                        else:
                            data_csv = config.csv_dir_syntest_est_sp_indoor
                    elif test_real_mode == 2:
                        # df = pd.read_csv('/data4/chengyean/data/outdoor_syn_test.csv')
                        if self.gt_test:
                            data_csv = config.csv_dir_syntest_gt_sp_outdoor
                        else:
                            data_csv = config.csv_dir_syntest_est_sp_outdoor
                    df = pd.read_csv(data_csv)
                    self.data_group_dirs = df['0'].tolist()
                else:
                    import pandas as pd
                    if test_real_mode == 0:
                        raise NotImplementedError
                    elif test_real_mode == 1:
                        # df = pd.read_csv('/data4/chengyean/data/indoor_syn_test.csv')
                        if self.gt_test:
                            data_csv = config.csv_dir_syntest_gt_face_indoor
                        else:
                            data_csv = config.csv_dir_syntest_est_face_indoor
                        df = pd.read_csv(data_csv)
                        self.data_group_dirs = df['0'].tolist()
                        self.data_group_dirs = [i.replace('Face-Light-v2-PredSphere', 'Face-Light-v2-PredFace') for i in self.data_group_dirs]
                    elif test_real_mode == 2:
                        if self.gt_test:
                            data_csv = config.csv_dir_syntest_gt_face_outdoor
                        else:
                            data_csv = config.csv_dir_syntest_est_face_outdoor
                        # df = pd.read_csv('/data4/chengyean/data/outdoor_syn_test.csv')
                        df = pd.read_csv(data_csv)
                        self.data_group_dirs = df['0'].tolist()
                        self.data_group_dirs = [i.replace('Face-Light-v2-PredSphere', 'Face-Light-v2-PredFace') for i in self.data_group_dirs]
                    elif test_real_mode == 3:
                        # df = pd.read_csv('/data4/chengyean/data/outdoor_syn_test.csv')
                        # self.data_group_dirs = df['0'].tolist()
                        dp = []
                        for path, _, _ in os.walk(config.visual_data_dir):
                            if 'albedo.hdr' in os.listdir(path):
                                dp.append(path)
                        self.data_group_dirs = dp
                        self.data_group_dirs.sort()
                        
                        self.compo_filename = config.compo_filename_face_finetune
                        data_csv = config.visual_data_dir
                        # df = pd.DataFrame(self.data_group_dirs)
                        # df.to_csv('./verbose/indoor_syn_visualize.csv', index=True)
                        # st()

                        # compo_filename_face_finetune visual_data_dir
                    
                    # st()
                print("Loading CSV: " + data_csv)

        # st()
        self.warp2sphere = warp2sphere
        # if not self.warp2sphere and not self.test_real:
        #     self.compo_filename = config.compo_filename_google

        # st()
        
        self.face_hw = face_hw
        self.sphere_hw = sphere_hw

        self.randaug = None
        self.google_net = google_net
        if self.google_net:
            self.compo_filename = config.compo_filename_google
        
        if self.warp2sphere:
            mask, normal = generate_sphere_mask_and_normal(out_hw=self.sphere_hw)
            self.mask_sp, self.normal_sp = mask, normal
            if self.train:
                self.randaug = RandAugment(n=2, m=10,
                                           augment_list=["exposure",
                                                         "white_balance",
                                                         "noise_sphere"])
        else:
            if self.train:
                self.randaug = RandAugment(n=2, m=10,
                                           augment_list=[
                                                         "exposure",
                                                         "white_balance",
                                                         "noise_face"])
        
        if self.ae and self.train:
            self.randaug = RandAugment(n=2, m=10, augment_list=["exposure", "white_balance"])

        
        # average time for each image
        self.avg_time = 0.0

        # debug
        if not use_aug:
            print("[FaceDatasetV2] No data augmentation.")
            self.randaug = None
        else:
            print("[FaceDatasetV2] Data augmentation.")

        # st()
        if not self.train:
            print("Testing number of images: {}".format(len(self.data_group_dirs)))
            
        if kwargs['kwargs'].rebuttal_3dv:
            self.rebutall_3dv = True
            self.compo_filename = {'face': 'face.hdr', 'envmap': 'envmap.hdr'}
            info = np.load('/userhome/feifan/dataset/Face-Light-v2/test/dataset-v2-info.pkl',allow_pickle=True)
            self.data_group_dirs = info['outdoor'] + info['sky']
            

    def __len__(self):
        return len(self.data_group_dirs)
    
    def setup_rerender_config(self):
        self.rerender_paths = {
            'input': 'SynValid_GT_Face',
            'face_compos': 'SynValid_EstBySynonly_Face',
            'albedo': 'SynValid'
        }
    
    def parse_albedo_path(self, basename):
        indexs = basename.split('-')
        first = indexs[1] + '-' + indexs[2]
        second = indexs[3] + '-' + indexs[4]
        return first + '/' + second, indexs[0]
        
    def prepare_rerender_batch(self, index):
        # breakpoint()
        prefix = 'SynValid_EstBySynonly_Sph'
        if 'SynValid_EstBySynonly_Sph' in self.data_group_dirs[index]:
            prefix = 'SynValid_EstBySynonly_Sph'
        elif 'SynValid_EstBySynonly_Face' in self.data_group_dirs[index]:
            prefix = 'SynValid_EstBySynonly_Face'
        elif 'SynValid_GT_Face' in self.data_group_dirs[index]:
            prefix = 'SynValid_GT_Face'
        
        input_face_path = self.data_group_dirs[index].replace(
            prefix, self.rerender_paths['input']
        )
        compos_face_path = self.data_group_dirs[index].replace(
            prefix, self.rerender_paths['face_compos']
        )
        albedo_path = self.data_group_dirs[index].replace(
            prefix, self.rerender_paths['albedo']
        )
            
        face_batch = {}
        face_batch['rerender_input'] = read_image(os.path.join(input_face_path, 
                                                               'face.png'))
        face_batch['rerender_normal'] = read_image(os.path.join(compos_face_path, 
                                                                'normal.png'))
        face_batch['rerender_normal'] = (face_batch['rerender_normal'] - 0.5) * 2
        
        face_batch['rerender_mask'] = read_mask(os.path.join(compos_face_path, 
                                                                'mask.png'))

        basename = os.path.basename(albedo_path)
        parsed_base_name, mark = self.parse_albedo_path(basename)
        parsed_albedo_path = albedo_path.replace(basename, parsed_base_name)
        if mark == 'S': 
            parsed_albedo_path = parsed_albedo_path.replace('outdoor', 'sky')
        
        face_batch['rerender_albedo'] = read_image(os.path.join(parsed_albedo_path, 
                                                                'albedo.hdr'))
        
        # convert to tensor
        for compo in face_batch.keys():
            face_batch[compo] = torch.from_numpy(face_batch[compo].transpose(2, 0, 1))
        return face_batch
            
    def __getitem__(self, index: int):
        """
        Reads a light-level data group, does data augmentation,
        converts to torch.Tensor, then returns it.
        
        Args:
            index: The index of the data group to be read and returned.

        Returns:
            If warp2sphere is True, returns: (h = w = `self.sphere_hw`)
                "mask": [1, h, w], sphere mask indicating which pixel is filled;
                "normal": [3, h, w], in [-1, 1] = unit normal vector;
                "shading": [3, h, w], warped diffuse shading;
                "specular": [3, h, w], warped specular;
                "envmap": [3, h, w], the mirror ball hdr envmap;
                "info": random parameters used in rendering this data group.
            If warp2sphere is False, returns: (h = w = `self.face_hw`)
                "mask": [1, h, w], face region mask;
                "face": [3, h, w], rendered full face;
                "albedo": [3, h, w], face reflectance;
                "normal": [3, h, w], in [-1, 1] = unit normal vector;
                "shading": [3, h, w], diffuse shading;
                "specular": [3, h, w], warped specular;
                "envmap": [3, 128, 128], the mirror ball hdr envmap;
                "info": random parameters used in rendering this data group.
        """
        sample = {}
        info = None

        if self.ae:
            file_path = os.path.join(self.data_group_dirs[index], 'envmap.hdr')
            img = read_image(file_path)

            img = cv2.resize(img, dsize=(self.sphere_hw, self.sphere_hw),interpolation=cv2.INTER_AREA)
            sample['envmap'] = torch.tensor(img.transpose(2, 0, 1).astype(np.float32))
            # sample['envmap'] = transforms.Resize((self.sphere_hw, self.sphere_hw))(img)
            # sample['envmap'] = sample['envmap'].numpy().transpose(1, 2, 0)
            # apply random data augmentation
            if self.randaug is not None:
                sample = self.randaug(sample, show_res=False)
            
            # sample['envmap'] = torch.from_numpy(sample['envmap'].transpose(2, 0, 1))
            return sample

        # start_time = time.time()

        if 'laval' in self.data_group_dirs[index]:
            compo_filename = self.compo_filename_laval
        else:
            compo_filename = self.compo_filename
        # st()
        for component, file_name in compo_filename.items():
        # for component, file_name in self.compo_filename.items():
            file_path = os.path.join(self.data_group_dirs[index], file_name)
            if file_name.endswith((".png", ".jpg", ".hdr", "exr")):  # image file
                # print(file_path)
                if component == 'envmap' and not os.path.exists(file_path) and (self.test_real_mode == 3 or self.in_the_wild):
                    # Placeholder for in-the-wild real data
                    img = read_image(config.place_holder_envmap_sp)
                
                elif component == 'envmap' and 'Laval' in file_path:
                    # Laval: one face per envmap senario
                    laval_envmap_name = file_path.split('/')[-5]
                    laval_envmap_path = file_path.replace('envmap_sp', laval_envmap_name)
                    img = read_image(laval_envmap_path)
                elif component == 'face' and not os.path.exists(file_path) and self.in_the_wild:
                    face_id = self.data_group_dirs[index].split('/')[-1]
                    face_name = face_id + '_input.png'
                    real_path = os.path.join(config.in_the_wild_visual_syn_real, face_name)
                    img = read_image(real_path)
                elif component == 'face' and not os.path.exists(file_path) and self.test_real:
                    sub_dir = self.data_group_dirs[index].split('/')[-2]
                    face_id = self.data_group_dirs[index].split('/')[-1]
                    face_name = face_id + '_input.png'
                    real_path = os.path.join(os.path.dirname(self.data_group_dirs[index].replace(sub_dir, sub_dir+'_visual')), face_name)
                    img = read_image(real_path)
                
                else:
                    # only for face / face_recon debug, can delete after 1102
                    img = read_image(file_path)
                
                if img is None:
                    raise Exception(f"Failed to read image file{file_name}.")

                if len(img.shape) == 2:  # mask
                    img = img[:, :, None]
                if component == "normal":  # make sure norm of normal vector is 1
                    img = img * 2 - 1  # use in normal.png, transform [0, 1] to [-1, 1]
                    img = img / np.sqrt(np.sum(img ** 2, axis=2, keepdims=True) + 1e-12)

                # img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
                sample[component] = img
            # st()
            if component == "info":  # txt file
                info = read_info(file_path)
            # sample['img_name'] = self.data_group_dirs[index].split('/')[-1]
        # st()
        if self.test_real:
            # map shading to 0-1 and apply the multiplier to specular
            multiplier = sample['shading'].max()
            sample['shading'] = sample['shading'] / multiplier
            sample['specular'] = sample['specular'] / multiplier
        # time1 = time.time() - start_time
        # ==================== BEGIN: Read from Sphere ====================

        if self.read_from_sphere:                
            for compo in sample.keys():
                if compo != "envmap":
                    if compo == "info" or type(sample[compo]) == str:
                        continue
                    sample[compo] = torch.from_numpy(sample[compo].transpose(2, 0, 1))
            # st()
            sample['envmap'] = cv2.resize(sample["envmap"], dsize=(self.sphere_hw, self.sphere_hw),interpolation=cv2.INTER_AREA)
            sample["envmap"] = torch.tensor(sample["envmap"].transpose(2, 0, 1).astype(np.float32))
            # sample["envmap"] = transforms.Resize((self.sphere_hw, self.sphere_hw))(sample["envmap"])

            sample['mask'] = sample['mask'][0, :, :][None, :, :]
            # st()
            # apply random data augmentation
            if self.randaug is not None:
                warped = self.randaug(sample, show_res=False)
            # add sphere normal
            sample["normal"] = self.normal_sp.astype(np.float32)
            sample["normal"] = torch.from_numpy(sample["normal"].transpose(2, 0, 1).astype(np.float32))
            # masking
            sample["shading"] = sample["shading"] * sample["mask"]
            sample["specular"] = sample["specular"] * sample["mask"]
            for compo in sample.keys():
                if type(sample[compo]) != str:
                    if sample[compo].shape[-1] != 64:
                        sample[compo] = transforms.Resize((64, 64))(sample[compo])

            sample['img_name'] = self.data_group_dirs[index].split('/')[-1]
            if self.rerender_loss:
                face_batch = self.prepare_rerender_batch(index)
                sample.update(face_batch)
            return sample

        # ==================== END: Read from Sphere ====================

        # ==================== BEGIN: Return Sphere ====================
        if self.warp2sphere:
            for compo in sample.keys():
                if compo == "info":
                    continue
                # sample[compo] = sample[compo].numpy().transpose(1, 2, 0)

            warped = warp_face_to_sphere(sample, out_hw=self.sphere_hw, verbose=False)
            # add envmap
            # warped["envmap"] = cv2.resize(sample["envmap"], dsize=(self.sphere_hw, self.sphere_hw),interpolation=cv2.INTER_AREA)
                                          
            for compo in warped.keys():
                if compo == "info":
                    continue
                warped[compo] = torch.from_numpy(warped[compo].transpose(2, 0, 1))
            
            sample['envmap'] = cv2.resize(sample["envmap"], dsize=(self.sphere_hw, self.sphere_hw),interpolation=cv2.INTER_AREA)
            warped["envmap"] = torch.tensor(sample["envmap"].transpose(2, 0, 1).astype(np.float32))
            # warped["envmap"] = transforms.Resize((self.sphere_hw, self.sphere_hw))(sample["envmap"])

            # apply random data augmentation
            # st()
            if self.randaug is not None:
                warped = self.randaug(warped, show_res=False)
            # add sphere normal
            warped["normal"] = self.normal_sp.astype(np.float32)
            warped["normal"] = torch.from_numpy(warped["normal"].transpose(2, 0, 1).astype(np.float32))
            # masking
            warped["shading"] = warped["shading"] * warped["mask"]
            warped["specular"] = warped["specular"] * warped["mask"]

            # if real data, keep the original face image for visualization
            # if self.test_real:
            if not self.train:
                for compo in ['shading', 'specular', 'face', 'normal', 'mask']:
                    if compo in sample.keys():
                        # interp = cv2.INTER_NEAREST if component in ["mask"] else cv2.INTER_LINEAR
                        # img = cv2.resize(sample[compo], dsize=(self.face_hw, self.face_hw), interpolation=interp)
                        img = torch.tensor(sample[compo].transpose(2, 0, 1))
                        img = transforms.Resize((self.face_hw, self.face_hw))(img)
                        if len(img.shape) == 2:
                            # img = img[:, :, None]
                            img = img[None, :, :]
                        warped[compo+'_face'] = img
                warped['env_name'] = self.data_group_dirs[index].split('/')[-3] # indoor-1, outdoor-2, indoor-3
                warped['img_name'] = os.path.basename(self.data_group_dirs[index])
                
            sample = warped
            del warped

        # ==================== END: Return Sphere ====================

        # ==================== BEGIN: Return Face ====================
        else:
            # st()
            # resize to self.face_hw
            for component, img in sample.items():
                if component != "envmap" and type(img) != str:
                    interp = cv2.INTER_NEAREST if component in ["mask"] else cv2.INTER_LINEAR
                    img = cv2.resize(img, dsize=(256, 256), interpolation=interp)
                    # img = cv2.resize(img, dsize=(self.face_hw, self.face_hw), interpolation=interp)
                    # st()
                    if len(img.shape) == 2:
                        img = img[:, :, None]
                    img = torch.tensor(img.transpose(2, 0, 1))
                    # st()
                        # img = img[None, :, :]
                    # img = transforms.Resize((self.face_hw, self.face_hw))(img)
                sample[component] = img
                
            # apply random data augmentation

            # CYA: resize envmap to (shpere size, sphere size)
            # for google net
            if self.google_net:
                sample["envmap"] = cv2.resize(sample["envmap"], dsize=(32, 32),interpolation=cv2.INTER_AREA)
            elif self.rec_envmap:
                sp_mask = generate_sphere_mask_from_size_np(sample["envmap"])
                # st()
                sample["envmap"] = warp_sp2rect(sample["envmap"], sp_mask, out_h=256)
            else:
                sample["envmap"] = cv2.resize(sample["envmap"], dsize=(64, 64),interpolation=cv2.INTER_AREA)
                # sample["envmap"] = torch.tensor(sample["envmap"].transpose(2, 0, 1).astype(np.float32))
            # for tdv net
            # sample["envmap"] = cv2.resize(sample["envmap"], dsize=(64, 64),interpolation=cv2.INTER_AREA)
            sample["envmap"] = torch.tensor(sample["envmap"].transpose(2, 0, 1).astype(np.float32))
            # sample["envmap"] = transforms.Resize((self.sphere_hw, self.sphere_hw))(sample["envmap"])
            # st()
            if self.randaug is not None:
                sample = self.randaug(sample, show_res=False)

            for component, img in sample.items():
                if component in ["face", "albedo"]:  # clip to [0, 1]
                    # img = np.clip(img, a_min=0, a_max=1)
                    img = torch.clip(img, 0, 1)
                # if component in ["normal", "albedo", "face", "shading", "specular"]:  # mask
                #     img = img * sample["mask"]
                sample[component] = img
            
            if self.test_real:
                # for compo in ['shading', 'specular', 'face', 'normal', 'mask']:
                # for compo in ['normal']:
                #     # interp = cv2.INTER_NEAREST if component in ["mask"] else cv2.INTER_LINEAR
                #     # img = cv2.resize(sample[compo], dsize=(self.face_hw, self.face_hw), interpolation=interp)
                    # img = sample[compo]
                    # img = transforms.Resize((self.face_hw, self.face_hw))(img)
                    # if len(img.shape) == 2:
                    #     # img = img[:, :, None]
                    #     img = img[None, :, :]
                    # sample[compo+'_face'] = img
                sample['env_name'] = self.data_group_dirs[index].split('/')[-3] # indoor-1, outdoor-2, indoor-3
                sample['img_name'] = os.path.basename(self.data_group_dirs[index])
            
            if self.rerender_loss:
                face_batch = self.prepare_rerender_batch(index)
                sample.update(face_batch)
            # st()
        # ==================== END: Return Face ====================

        # time2 = time.time() - start_time - time1
        # self.avg_time1 += time1
        # self.avg_time2 += time2

        # for component, img in sample.items():
        #     img = TF.to_tensor(img)
        #     sample[component] = img

        # sample["info"] = info

        # sample.pop("info")
        # print(f"Read data group {index} in {time1:.3f}s, warp to sphere in {time2:.3f}s.")
        #verbose: check component type and size
        # for compo in sample.keys():
        #     print(compo, type(sample[compo]), sample[compo].shape)
        # st()
        return sample
       
        
if __name__ == '__main__':
    # dataset = FaceDatasetV2(train=True, warp2sphere=False, sphere_hw=64, face_hw=64)
    # dataset = FaceDatasetV2(train=True, warp2sphere=False, sphere_hw=64, face_hw=64, test_real=False, use_aug=False)

    # dataset = FaceDatasetV2(train=True, warp2sphere=False, face_hw=64, sphere_hw=64, use_aug=True, env_datasets=None)
    # dataset = FaceDatasetV2(train=False, warp2sphere=True, face_hw=64, sphere_hw=64, use_aug=0, test_real=True, test_real_mode=0)
    
    # dataset = FaceDatasetV2(train=True, warp2sphere=False, face_hw=64, sphere_hw=64, use_aug=1, google_net=False, rect_envmap=True)
    
    dataset = FaceDatasetV2(train=False, warp2sphere=True, sphere_hw=64, face_hw=64, use_aug=0, test_real=0, test_real_mode=2)
    # dataset = FaceDatasetV2(train=True, warp2sphere=True, sphere_hw=64, use_aug=1, env_datasets=['outdoor', 'sky'], read_from_sphere=1, finetune=False)
    # dataset = FaceDatasetV2(train=True, warp2sphere=False, sphere_hw=64, use_aug=1, env_datasets=None, read_from_sphere=False)
    # dataset = FaceDatasetV2(train=True, warp2sphere=True, sphere_hw=64, face_hw=128, use_aug=1, read_from_sphere=1)
    out_dir_name = "dataset-v2-sphere"
    
    # dataset = FaceDatasetV2(train=True, warp2sphere=False, face_hw=512)
    # out_dir_name = "dataset-v2-face"
    print(f"\n{len(dataset)} groups of training data in total.\n")
    
    batch_size = 128
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0)
    # dataloader = DataLoader(dataset,
    #                         batch_size=batch_size,
    #                         num_workers=32,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         drop_last=True)
    
    epoch_num = 2
    num_batches = 1
    
    import time

    for epoch in range(epoch_num):
        start_time = time.time()
        meta_max = 0
        meta_min = 1e8
        for i, batch in enumerate(tqdm(dataloader)):
            st()
            # print max of batch
            max_i = 0
            min_i = 1e8
            for key in batch.keys():
                if key == "info" or key == 'envmap' or key == 'normal':
                    continue
                max_i = max(max_i, torch.max(batch[key]))
                min_i = min(min_i, torch.min(batch[key]))
            meta_max = max(meta_max, max_i)
            meta_min = min(meta_min, min_i)
        print(f"max: {meta_max}, min: {meta_min}")
        # print(f'batch {i} max {max_i}')
            # pass
        end_time = time.time()
        print(f"Epoch {epoch} takes {end_time - start_time} seconds.")
        # torchvision.utils.save_image(batch['envmap'][0, ...], 'verbose/rect.png')