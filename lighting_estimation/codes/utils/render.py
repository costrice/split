# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from skimage import color
from tqdm import tqdm

import config
from utils.envmap import collect_all_envmaps
from utils.general import check_light_level, linrgb2srgb, read_image, write_image
from utils.mesh import collect_all_3d_faces

from pdb import set_trace as st

def collect_rendered(train: bool,
                     load: bool = True,
                     dataset_dir: str = None, 
                     compo_dict: dict = {}):
    """Collect rendered dataset.
    
    Args:
        train: whether collects training set or testing set.
        load: whether tries to load existing dataset information.
        dataset_dir: path to the rendered dataset.
        
    Returns:
        Dict[str, List[str]]: a dict containing paths to all valid data groups
            separated by envmap dataset used.
    """
    phase = "train" if train else "test"
    if dataset_dir is None:
        dataset_dir = config.dir_rendered_dataset[phase]
    # st()
    dataset_info_file = os.path.join(dataset_dir, "dataset-v2-info.pkl")
    if load and os.path.exists(dataset_info_file):
        # try to load
        print(f"\nLoad dataset information from {dataset_info_file}.")
        with open(dataset_info_file, "rb") as f:
            groups_dict = pickle.load(f)
        print(f"\nFind {[len(groups_dict[dataset]) for dataset in config.envmap_dataset_names]} "
              f"valid face-light groups for {config.envmap_dataset_names} respectively.")
        return groups_dict
    
    # if info not loaded, collect data
    groups_dict = {}  # complete group
    incomplete_groups = []
    
    for dataset in config.envmap_dataset_names:
        # collect face-level directories
        groups_dict[dataset] = []
        face_groups = sorted(filter(lambda entry: entry.is_dir() and config.face_group_regex.match(entry.name),
                                    os.scandir(os.path.join(dataset_dir, dataset))),
                             key=lambda entry: entry.name)
        for face_group in tqdm(face_groups):
            light_groups = sorted(filter(lambda entry: entry.is_dir() and
                                                       config.light_group_regex.match(entry.name),
                                         os.scandir(face_group.path)),
                                  key=lambda entry: entry.name)
            for light_group in light_groups:
                if not check_light_level(light_group.path, just_face=False, just_light=False, verbose=True, compo_dict=compo_dict):
                    incomplete_groups.append(light_group.path)
                    continue
                groups_dict[dataset].append(light_group.path)
    
    print(f"\nCollecting complete.")
    print(f"\nFind {[len(groups_dict[dataset]) for dataset in config.envmap_dataset_names]} "
          f"valid face-light groups for {config.envmap_dataset_names} respectively.")
    print(f"\nFind {len(incomplete_groups)} incomplete data groups:\n ",
          "\n  ".join(incomplete_groups))
    
    print(f"Save dataset information into {dataset_info_file}.")
    with open(dataset_info_file, "wb") as f:
        pickle.dump(groups_dict, f, pickle.HIGHEST_PROTOCOL)
    
    return groups_dict


def generate_random_params(train: bool):
    """Generate random parameters (e.g. poses) used in rendering faces, then save it as a file.
    
    Args:
        train: if True, generate random parameters for training (otherwise for testing).
    """
    phase = "train" if train else "test"
    output_path = config.dir_rendered_dataset[phase]
    
    rng = np.random.default_rng(seed=233 if train else 2333)
    
    # collect meshes and(or) envmaps
    faces = collect_all_3d_faces(train=train, frontal=True, verbose=False)
    pose_per_face = config.poses_per_face[phase]
    
    print(f"\nGenerating random parameters for {phase}ing set... ")
    
    # =============== BEGIN: Generate random parameters for face pose ===============
    # initialization
    focal_length = np.zeros((len(faces), pose_per_face), dtype=np.float32)
    face_distance = np.zeros((len(faces), pose_per_face), dtype=np.float32)
    face_rotation = np.zeros((len(faces), pose_per_face, 3), dtype=np.float32)
    
    focal_length_list = [setting[0] for setting in config.camera_settings]
    face_distance_list = [setting[1] for setting in config.camera_settings]
    weight_list = [setting[2] for setting in config.camera_settings]
    weight_list = np.array(weight_list, dtype=np.float32)
    weight_list /= np.sum(weight_list)
    
    for i in tqdm(range(len(faces))):
        for j in range(config.poses_per_face[phase]):
            # camera setting (focal length and corresponding face distance)
            setting_idx = rng.choice(len(focal_length_list), p=weight_list)
            focal_length[i, j] = focal_length_list[setting_idx]
            face_distance[i, j] = face_distance_list[setting_idx] * \
                                  rng.uniform(low=config.face_distance_interval[0],
                                              high=config.face_distance_interval[1])
            # face rotation (pitch, yaw, roll)
            pitch = config.face_pose_pitch * rng.normal(0, 0.4)
            yaw = config.face_pose_yaw * rng.normal(0, 0.4)
            roll = config.face_pose_roll * rng.normal(0, 0.4)
            face_rotation[i, j] = [pitch, yaw, roll]
    
    # =============== END: Generate random parameters for face pose ===============
    
    # =============== BEGIN: Generate random parameters for envmaps ===============
    # collect envmaps
    envmaps = collect_all_envmaps(train=train)
    print(f"\n{len(envmaps)} envmap datasets found. "
          f"{config.envmaps_per_pose} envmaps will be used for dataset {config.envmap_dataset_names}"
          f" for every pose respectively.")
    
    envmap_index = {}
    envmap_spin = {}
    specular_roughness = {}
    specular_intensity = {}
    displacement_scale = {}
    
    for dataset, amount in zip(config.envmap_dataset_names, config.envmaps_per_pose):
        # initialize
        envmap_index[dataset] = np.zeros((len(faces), pose_per_face, amount), dtype=np.int32)
        envmap_spin[dataset] = np.zeros((len(faces), pose_per_face, amount), dtype=np.float32)
        specular_roughness[dataset] = np.zeros((len(faces), pose_per_face, amount), dtype=np.float32)
        specular_intensity[dataset] = np.zeros((len(faces), pose_per_face, amount), dtype=np.float32)
        displacement_scale[dataset] = np.zeros((len(faces), pose_per_face, amount), dtype=np.float32)
        
        for i in tqdm(range(len(faces))):
            for j in range(config.poses_per_face[phase]):
                for k in range(amount):
                    envmap_index[dataset][i, j, k] = rng.choice(len(envmaps[dataset]))
                    envmap_spin[dataset][i, j, k] = rng.uniform(low=-config.envmap_spin,
                                                                high=config.envmap_spin)
                    rfns = rng.normal(loc=config.spec_rfns_mean, scale=config.spec_rfns_std)
                    rfns = np.clip(rfns,
                                   a_min=config.spec_rfns_mean - 2.5 * config.spec_rfns_std,
                                   a_max=config.spec_rfns_mean + 2.5 * config.spec_rfns_std)
                    specular_roughness[dataset][i, j, k] = rfns
                    specular_intensity[dataset][i, j, k] = rng.uniform(*config.spec_intensity_interval)
                    displacement_scale[dataset][i, j, k] = rng.uniform(*config.displacement_scale_interval)
    
    random_params_dict = {
        "focal_length": focal_length,
        "face_distance": face_distance,
        "face_rotation": face_rotation,
        "envmap_idx": envmap_index,
        "envmap_spin": envmap_spin,
        "specular_roughness": specular_roughness,
        "specular_intensity": specular_intensity,
        "displacement_scale": displacement_scale
    }
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(os.path.join(output_path, "random_params.npy"), random_params_dict)
    
    print("Random number generation finished.")


def load_generated_params(train: bool):
    """Load random parameters generated beforehand.
    
    Args:
        train: if True, load parameters for training (otherwise for testing)
        
    Returns:
        random_params, which are generated random parameters.
    """
    phase = "train" if train else "test"
    output_path = config.dir_rendered_dataset[phase]
    random_params = np.load(os.path.join(output_path, "random_params.npy"), allow_pickle=True).item()
    return random_params


def get_mask(albedo_srgb: np.ndarray,
             hair: bool = True,
             bathing_cap: bool = True):
    """Generate mask for bathing cap and (or) skin hair.
    
    Args:
        albedo_srgb: the albedo according to which the hair and bathing cap are identified.
        hair: whether mask out hair.
        bathing_cap: whether mask out red bathing cap.
        
    Returns:
        A mask, 0 means no obstruction to skin, 1 means yes.
    """
    # transform to YCbCr color space
    albedo_YCbCr = color.rgb2ycbcr(albedo_srgb)
    Y, Cb, Cr = [albedo_YCbCr[:, :, c] for c in range(3)]
    # generating mask
    mask = Y < 0  # All False
    if hair:
        mask += (Y <= 85) * (Cb > 118) * (Cb < 140) * (Cr > 118) * (Cr < 140)  # black hair
    if bathing_cap:
        # yellow bathing cap (not applicable)
        # mask += (Y >= 140) * (Y < 170) * (Cb > 100) * (Cb < 110) * (Cr > 140) * (Cr < 150)
        # red bathing cap
        mask += (Y <= 120) * (Cb > 110) * (Cb < 130) * (Cr > 170) * (Cr < 230)
    return mask


def post_process(data_dir: str,
                 verbose: bool = True):
    """Does post processing for a directory, which is masking out possibly existing red bathing cap.
    
    Args:
        data_dir: the path to the data group directory.
        verbose: if True, outputs some more message.
        
    Returns:
        bool: True indicats process completed, False indicates process skipped or cannot be done.
    """
    if not check_light_level(data_dir, just_face=True, verbose=verbose):
        print(f"Face part for data group {data_dir} is incomplete. Skip its post processing.")
        return False
    
    # ==================== BEGIN: masking out bathing cap ====================
    mask = read_image(os.path.join(data_dir, config.compo_filename["mask"]))
    albedo = read_image(os.path.join(data_dir, config.compo_filename["albedo"]))
    idx_h = np.repeat(np.arange(mask.shape[0])[:, None], mask.shape[1], axis=1)
    h_min = np.min(mask * idx_h + (1 - mask) * mask.shape[0])
    h_max = np.max(mask * idx_h)
    bathing_cap_mask = get_mask(linrgb2srgb(albedo), hair=False, bathing_cap=True).astype(np.uint8)
    bathing_cap_mask *= (idx_h < (h_min + (h_max - h_min) * 0.3))
    mask = mask * (1 - bathing_cap_mask)

    # saving results
    write_image(os.path.join(data_dir, config.compo_filename["mask"]), mask)
    # ==================== END: masking out bathing cap ====================
    return True


def post_process_normal(train):
    rendered = collect_rendered(train=train, load=True)
    
    for dataset in config.envmap_dataset_names:
        dirs = rendered[dataset]
        for data_dir in tqdm(dirs):
            if not os.path.exists(os.path.join(data_dir, "normal.png")):
                normal = read_image(os.path.join(data_dir, "normal.exr"))
                mask = read_image(os.path.join(data_dir, "mask.png"))[:, :, None]
                write_image(os.path.join(data_dir, "normal.png"), (normal + 1) / 2 * mask, depth=16)


def delete_exr(train):
    rendered = collect_rendered(train=train, load=True)
    
    for dataset in config.envmap_dataset_names:
        dirs = rendered[dataset]
        for data_dir in tqdm(dirs):
            if os.path.exists(os.path.join(data_dir, "normal.png")) and \
                    os.path.exists(os.path.join(data_dir, "normal.exr")):
                os.remove(os.path.join(data_dir, "normal.exr"))


if __name__ == '__main__':
    # generate_random_params(train=True)
    # random_params = load_generated_params(train=True)
    
    # generate_random_params(train=False)
    
    # post_process_for_shared(data_dir="D:\\AlbedoRecon\\example\\013-20-01", verbose=True)
    
    # normal = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.exr")
    # normal_png = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.png")
    # mask = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\mask.png")[:, :, None]
    # write_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.png", (normal + 1) / 2 * mask, depth=16)
    
    # normal = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.exr")
    # normal_png = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.png")
    # mask = read_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\mask.png")[:, :, None]
    # diff = normal * mask - (normal_png * 2 - 1) * mask
    # print(np.sum(diff ** 2) / np.sum(normal_png ** 2))
    # write_image(r"E:\dataset\Face-Light-v2\train\indoor\001-01\01-01\normal.png", (normal + 1) / 2 * mask, depth=16)
    
    # post_process_normal(train=True)
    
    # collect_rendered(train=False, load=False)
    
    # post_process_normal(train=False)
    
    # collect_rendered(train=True, load=False)
    #
    
    # delete_exr(train=True)
    # delete_exr(train=False)
    
    pass