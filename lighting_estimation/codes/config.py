# -*- coding: utf-8 -*-

import os
import numpy as np
import re

from sys import platform

# Dataset type
# dataset_type = 'albedo'
dataset_type = 'face'

# training_type
# linet: pipeline 0
# autoencoder: autoencoder in pipeine 1
# predictor: predictor in pipeline 1

# Need List
need_ls = need_ls_sp = ['normal', 'envmap', 'shading', 'specular', 'segment']

need_env_ls = ['envmap']

# input scale for face
input_face_scale = 64
# input_scale = input_face_scale = 512
output_scale = sphere_scale = 64
input_sphere_scale = 128
latent_scale = 1024

scheduler_step = [15, 30, 45]
ae_scheduler_step = [50, 100, 150]

use_aug = False

loss_ratio = [0.6, 0.2, 0.2] # loss for Hight freq, Low freq, Mid freq
# loss_ratio = [0.2, 0.2, 0.6] # loss for Hight freq, Low freq, Mid freq

# Multiscale Supervision
MULTISCALE = True
DIS_MULTISCALE = True

# Albedo as input
ALBEDO_TRAINING = False
NORMAL_TRAINING = True
LICOL_TRAINING = False
SHADING_TRAINING = True
SPECULAR_TRAINING = True
FACE_TRAINING = False
# training_need_ls = ['face', 'albedo', 'normal', 'light_color', 'shading', 'specular']
training_need_ls = training_need_ls_sp = ['normal','shading', 'specular','mask']
# order matters! 
training_need_ls_google = ['face', 'mask']

# ==================== BEGIN: Data Composition and Naming ====================
# Data Folder Naming
person_regex = re.compile("^[0-9]+$")  # person id
expr_regex = re.compile("^[0-9]+_[a-zA-Z_]+$")  # expression
face_group_regex = re.compile("^[0-9]{3}-[0-9]{2}$")  # rendered face-level directory
light_group_regex = re.compile("^[0-9]{2}-[0-9]{2}$")  # rendered light-level directory

# Rendered Data Composition
# rendered group prior to all post-processing
compo_filename = {
    "mask": "mask.png",
    "normal": "normal.png",
    # "albedo": "albedo.hdr",  # diffuse albedo
    "face": "face.hdr",
    "shading": "shading.hdr",  # diffuse shading
    "specular": "specular.hdr",
    "envmap": "envmap.hdr",
    # "info": "info.txt",
}
compo_filename_google = {
    "mask": "mask.png",
    # "normal": "normal.png",
    # "albedo": "albedo.hdr",  # diffuse albedo
    "face": "face.hdr",
    # "shading": "shading.hdr",  # diffuse shading
    # "specular": "specular.hdr",
    "envmap": "envmap.hdr",
    # "info": "info.txt",
}

compo_filename_sphere = {
    "mask": "mask.hdr",
    "shading": "shading.hdr",
    "specular": "specular.hdr",
    "envmap": "envmap.hdr",
}

compo_filename_face_finetune = {
    "mask": "mask.png",
    "shading": "shading.hdr",
    "specular": "specular.hdr",
    "envmap": "envmap.hdr",
    'normal': 'normal.png',
    "face": 'face.png'
    # "face": 'face.hdr'
}

compo_filename_sphere_finetune = {
    "mask": "mask.png",
    "shading": "shading.hdr",
    "specular": "specular.hdr",
    "envmap": "envmap.hdr",
}

compo_real_data = {
    "mask": "mask.png",
    "normal": "normal.png",
    # "albedo": "albedo.hdr",  # diffuse albedo
    # "face": "face_recon.hdr",
    "face": "face.hdr",
    "shading": "shading.hdr",  # diffuse shading
    "specular": "specular.hdr",
    "envmap": "../../envmap_proc/envmap_sp.hdr",
    # "envmap": "../../envmap_proc/envmap.hdr",
    # "info": "info.txt",
}


compo_real_data_inthewild = {
    "mask": "mask.png",
    "normal": "normal.png",
    # "albedo": "albedo.hdr",  # diffuse albedo
    # "face": "face_recon.hdr",
    "face": "face.hdr",
    "shading": "shading.hdr",  # diffuse shading
    "specular": "specular.hdr",
    "envmap": "../../envmap_proc/envmap_sp.hdr",
    # "info": "info.txt",
}

compo_real_data_laval = {
    "mask": "mask.png",
    "normal": "normal.png",
    # "albedo": "albedo.hdr",  # diffuse albedo
    "face": "face_recon.hdr",
    "shading": "shading.hdr",  # diffuse shading
    "specular": "specular.hdr",
    "envmap": "envmap_sp.hdr",
    # "info": "info.txt",
}
# ==================== END: Data Composition and Naming ====================



# ==================== BEGIN: Data Folder and Path ====================
dir_code = os.path.dirname(__file__)  # dir of this file
dir_proj = os.path.dirname(dir_code)  # dir of the whole project
dir_ckpts = os.path.join(dir_proj, "checkpoints")  # dir of network checkpoints
dir_data_files = os.path.join(dir_proj, "data_files")  # dir of useful data files
# different path for different platform
if platform == "linux":
    # dir_datasets = "/data4/chengyean/data/"
    dir_datasets = "/userhome/chengyean/face_lighting/data/"
    dir_examples = "/userhome/feifan/AlbedoRecon/example"
    base_path_output = os.path.join(dir_datasets, "dataset_v2")
elif platform == "win32":
    dir_datasets = r"D:\dataset"
    dir_examples = r"D:\AlbedoRecon\example"
    base_path_output = os.path.join(r"E:\dataset", "Face-Light-v2")
else:
    raise ValueError(f"Unknown platform: {platform}")
# 3d face dataset
base_dir_3d_face = os.path.join(dir_datasets, "FaceScape")
# envmap datasets
envmap_dataset_names = ["indoor", "outdoor", "sky"]
base_dirs_envmap_datasets = {
    "indoor": os.path.join(dir_datasets, "IndoorHDRDataset2018"),
    "outdoor": os.path.join(dir_datasets, "outdoorPanosExr"),
    "sky": os.path.join(dir_datasets, "LavalSkyHDR"),
}

dir_3d_face_dataset = {}
dir_envmap_datasets = {}
dir_rendered_dataset = {}

dir_sphere_dataset = os.path.join(dir_datasets, "dataset_v2_sphere")
dir_rendered_dataset_sphere = {}

dir_finetune_dataset = os.path.join(dir_datasets, "Face-Light-v2-PredSphere")
dir_rendered_dataset_finetune = {}

dir_finetune_dataset_face = os.path.join(dir_datasets, "Face-Light-v2-PredFace")
dir_rendered_dataset_finetune_face = {}

for phase in ["train", "test"]:
    dir_3d_face_dataset[phase] = os.path.join(base_dir_3d_face, phase)
    
    dir_envmap_datasets[phase] = {}
    for dataset in envmap_dataset_names:
        dir_envmap_datasets[phase][dataset] = os.path.join(base_dirs_envmap_datasets[dataset], phase)
        
    dir_rendered_dataset[phase] = os.path.join(base_path_output, phase)
    dir_rendered_dataset_sphere[phase] = os.path.join(dir_sphere_dataset, phase)
    dir_rendered_dataset_finetune[phase] = os.path.join(dir_finetune_dataset, phase)
    dir_rendered_dataset_finetune_face[phase] = os.path.join(dir_finetune_dataset_face, phase)

brdf_dir = './BRDF_base'

real_data_full = ['indoor-1', 'indoor-2', 'indoor-3', 'indoor-4', 'indoor-5', 'indoor-6', 'indoor-7', 'indoor-8', 'indoor-9', 'outdoor-1', 'outdoor-2', 'outdoor-3', 'outdoor-4', 'outdoor-5', 'outdoor-6', 'outdoor-7', 'outdoor-8', 'outdoor-9']
real_data_indoor = ['indoor-1', 'indoor-2', 'indoor-3', 'indoor-4', 'indoor-5', 'indoor-6', 'indoor-7', 'indoor-8', 'indoor-9']
real_data_outdoor = ['outdoor-1', 'outdoor-2', 'outdoor-3', 'outdoor-4', 'outdoor-5', 'outdoor-6', 'outdoor-7', 'outdoor-8', 'outdoor-9']
real_data_prefix = 'face_decom_final_cbn'

# # real data
csv_dir_realtest_old_syn_real = 'lighting_estimation/test_list/full_real_testset_list_old_Syn+Real.csv'
csv_dir_realtest_old_syn_real_v2 = csv_dir_realtest_old_syn_real.replace('.csv', '_v2.csv')
# # final outdoor dataset: with new captured (2022.11) data and Laval data
csv_dir_laval_syn_real = 'lighting_estimation/test_list/laval_cvpr_Syn+Real_v2.csv'
csv_dir_laval_syn_real_v2 = csv_dir_laval_syn_real.replace('.csv', '_v2.csv')

# ==================== END: Data Folder Path ====================



# ==================== BEGIN: Data Content Setting ====================
# render how many times each face with different poses
poses_per_face = {"train": 4, "test": 3}
# render how many times each pose with different datasets of envmaps,
#   each number corresponding to 1 dataset in order.
# 5 indoor, 2 outdoor, 3 sky
envmaps_per_pose = {"indoor": 5, "outdoor": 2, "sky": 3}


face_pose_pitch = 35 / 180 * np.pi
face_pose_yaw = 35 / 180 * np.pi
face_pose_roll = 45 / 180 * np.pi

envmap_spin = 100 / 180 * np.pi

envmap_sphere_size = 64  # 64 x 64 sphere

# Random Camera Parameters. Each tuple contains:
#   focal length of perspective camera (mm);
#   face-camera distance (m);
#   random weight of this setting;
camera_settings = [(35, 2.3, 0.25),
                   (45, 2.9, 0.5),
                   (55, 3.4, 1),
                   (70, 4.2, 2),
                   (85, 5.1, 4),
                   (100, 6, 2),
                   (120, 7.2, 1),
                   (150, 9, 0.5),
                   (200, 12, 0.25)]

# Random Distance Parameters
face_distance_interval = [0.9, 1.1]

# Random Face Shader Settings
spec_rfns_mean = 0.45
spec_rfns_std = 0.04
spec_rfns_preset = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
spec_intensity_interval = [0.3, 1.2]
displacement_scale_interval = [0.4, 0.65]
# ==================== END: Data Content Setting ====================