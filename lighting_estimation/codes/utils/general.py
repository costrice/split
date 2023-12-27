# -*- coding: utf-8 -*-
import os
from typing import Dict

import cv2
import numpy as np

import config


def tensor2ndarray(img):
    """
    Convert a tensor to a numpy ndarray. The tensor must not contain batch dimension.
    """
    return np.transpose(img.cpu().detach().numpy(), (1, 2, 0))


def linrgb2srgb(color_linrgb):
    """
    Transform a image in [0, 1] from linear RGB to sRGB space.
    """
    big = color_linrgb > 0.0031308
    color_srgb = big * (1.055 * (color_linrgb ** (1 / 2.4)) - 0.055) + \
                 (~big) * color_linrgb * 12.92
    # color_srgb = color_linrgb ** (1 / 2.2)
    return color_srgb


def srgb2linrgb(color_srgb):
    """
    Transform a image in [0, 1] from sRGB to linear RGB space.
    """
    big = color_srgb > 0.0404482362771082
    color_linrgb = big * (((color_srgb + 0.055) / 1.055) ** 2.4) + \
                   (~big) * (color_srgb / 12.92)
    return color_linrgb


def convert_to_uint8(img: np.ndarray):
    """
    Convert an image into np.uint8. If float, clip to [0, 1] first.
    """
    if img.dtype == np.bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 255).astype(np.uint8)
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, a_min=0, a_max=1) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def convert_to_uint16(img: np.ndarray):
    """
    Convert an image into np.uint16. If float, clip to [0, 1] first.
    """
    if img.dtype == np.bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 255
    if img.dtype == np.uint16:
        return img
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, a_min=0, a_max=1) * 65535).astype(np.uint16)
    raise ValueError(f"Unsupported dtype: {img.dtype}")
    

def convert_to_float32(img: np.ndarray):
    """
    Convert an image into np.float32
    """
    if img.dtype == np.bool:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535
    if img.dtype == np.float32:
        return img
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def read_mask(mask_path) -> np.ndarray:
    """Read a mask image and make its shape (h, w, 1) and its type float."""
    # if isinstance(mask_path, Path):
    #     mask_path = str(mask_path)
    mask_img = read_image(mask_path)
    if len(mask_img.shape) == 2:
        mask_img = mask_img[:, :, None]
    else:
        mask_img = mask_img[:, :, :1]
    mask_img = (mask_img > 0.5).astype(float)
    return mask_img


def read_image(img_path: str):
    """
    read an image and convert to np.float32 (range in [0, 1] if image is LDR).
    
    Args:
        img_path: path to the image.
        
    Returns:
        ndarray, dtype=np.float32
    """
    if os.path.exists(img_path):
        img = cv2.imread(img_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f'ValueError: Failed to read image {img_path}')
            return None
        if len(img.shape) == 3:  # BGR
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = convert_to_float32(img)
        return img
    else:
        raise FileNotFoundError(f"File {img_path} not found.")


def write_image(output_path: str, img: np.ndarray, depth: int = 8):
    """
    Write an ndarray <img> into <output_path>, LDR or HDR according to extension specified in <output_path>.
    
    Args:
        output_path: the save path.
        img: the image to be saved.
        depth: bit depth, can be 8 or 16.
    """
    if img.dtype in [np.float64, np.bool]:  # cv2 do not support float64?
        img = img.astype(np.float32)
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
    if output_path.endswith((".hdr", ".exr")):
        cv2.imwrite(output_path, convert_to_float32(img))
    elif output_path.endswith((".png", ".jpg")):
        if depth == 8:
            cv2.imwrite(output_path, convert_to_uint8(img))
        elif depth == 16:
            cv2.imwrite(output_path, convert_to_uint16(img))
        else:
            raise ValueError(f"Unexpected depth {depth}")
    else:
        raise ValueError(f"Unexpected file extension in {output_path}")


def check_face_level(dirs: Dict[str, str], train: bool, verbose=False):
    """
    Check soundness (existence of all light-level groups) of data group in face-level <folders>.
    
    Args:
        dirs: A dict, containing paths to face-level dir for each envmap dataset.
        train: bool. Different pose amount are used for training and testing dataset.
        verbose: whether this prints info message.
        
    Returns:
        True if sound, else False.
    """
    phase = "train" if train else "test"
    for dataset, directory in dirs.items():
        for pose_id in range(config.poses_per_face[phase]):
            for light_id in range(config.envmaps_per_pose[dataset]):
                if not check_light_level(os.path.join(directory, f"{pose_id + 1:02d}-{light_id + 1:02d}"),
                                         verbose=verbose):
                    return False
    return True


def check_light_level(directory, just_face=False, just_light=False, verbose=False, compo_dict={}):
    """
    Checks if all specified (see <just_face> and <just_light>) data components
    are existing in light-level group <dir>.
    
    Args:
        directory: the light-level directory where existence of components are checked.
        just_face: if True, skip checking envmap.
        just_light: if True, only check envmap.
        verbose: whether this prints info message.
        
    Returns:
        True if all existing, else False.
    """
    # compo_dict = config.compo_filename
    # if read_from_sphere:
    #     compo_dict = config.compo_filename_sphere
    for component, file_name in compo_dict.items():
        if just_face and component in ["envmap"]:
            continue
        if just_light and component not in ["envmap"]:
            continue
        if not os.path.exists(os.path.join(directory, file_name)):
            if verbose:
                print(f"Component <{component}>: {file_name} is missing in group dir {directory}. ")
            return False
    return True


if __name__ == '__main__':
    data_dir = "D:\\AlbedoRecon\\example\\013-20-01"
    
    random_srgb = np.random.uniform(low=0, high=1, size=(400, 400, 3))
    random_linrgb = srgb2linrgb(random_srgb)
    random_srgb_recon = linrgb2srgb(random_linrgb)
    write_image(os.path.join(data_dir, "random_srgb.png"), random_srgb)
    write_image(os.path.join(data_dir, "random_srgb_recon.png"), random_srgb_recon)
    write_image(os.path.join(data_dir, "random_linrgb.png"), random_linrgb)
    
    albedo = read_image(os.path.join(data_dir, "albedo.hdr"))
    albedo = np.clip(albedo, a_min=0, a_max=1)
    write_image(os.path.join(data_dir, "albedo.png"), linrgb2srgb(albedo))
    
    albedo_png = cv2.imread(os.path.join(data_dir, "albedo.png"))
    
    envmap = read_image(os.path.join(data_dir, "9C4A0003-e05009bcad.exr"))
    envmap = np.clip(envmap * 50, a_min=0, a_max=1)
    write_image(os.path.join(data_dir, "envmap.png"), cv2.createTonemapReinhard(1.5, 0, 0, 0).process(envmap))