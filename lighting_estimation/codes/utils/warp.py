import os
import time
from typing import Dict

import cv2
import numpy as np

from utils.general import read_image, write_image

from pdb import set_trace as st

def warp_face_to_sphere(face_compos: Dict[str, np.ndarray],
                        out_hw: int = 512,
                        verbose: bool = False):
    """
    Warp components of a face into sphere representation, using its normal.
    
    Args:
        face_compos: a dict containing the component of a face as np.ndarray.
            Should include
            "normal": [h, w, 3], ranging in [-1, 1], representing the surface normal of each pixel,
            "mask": [h, w, 1], ranging in [0, 1], representing whether a pixel is in face region,
            "shading", "specular": the component to be warped.
        out_hw: the width and height of output sphere.
        verbose: if True, print time used.
        
    Returns:
        Dict[str, np.ndarray]: a dictionary containing the warped input component. Includes
            "shading", "specular": corresponding warped component of shape (out_hw, out_hw, 3).
            "mask": a mask indicating whether a pixel is occupied by a warped pixel in the original image.
    """
    normal = face_compos["normal"]
    mask = face_compos["mask"]
    shading = face_compos["shading"]
    specular = face_compos["specular"]
    
    # initialize warped sphere
    shading_sp = np.zeros((out_hw, out_hw, 3), dtype=np.float32)
    specular_sp = np.zeros((out_hw, out_hw, 3), dtype=np.float32)
    mask_sp = np.zeros((out_hw, out_hw, 1), dtype=np.float32)
    
    # ========== Fast Version ========== (~0.021 sec for 1 image)
    # start_time = time.time()
    # collect position and normal of face region pixel
    orig_x, orig_y, _ = np.where(mask > 0.5)  # (n_pix, ), (n_pix, )
    orig_normal = normal[orig_x, orig_y]  # (n_pix, 3) representing (x, y, z)
    # compute position in warped image
    warped_x = (np.trunc(-orig_normal[:, 1] * out_hw / 2 + out_hw / 2 - 1e-6)).astype(np.uint32)
    warped_y = (np.trunc(orig_normal[:, 0] * out_hw / 2 + out_hw / 2 - 1e-6)).astype(np.uint32)
    warped_x = np.clip(warped_x, 0, out_hw - 1)
    warped_y = np.clip(warped_y, 0, out_hw - 1)
    
    
    # allocate original values
    mask_sp[warped_x, warped_y] = 1
    # implement maximum by sorting then allocating
    shading_values = shading[orig_x, orig_y]
    sort_idx = np.argsort(shading_values[:, 1], axis=0)  # from dim to bright
    shading_sp[warped_x[sort_idx], warped_y[sort_idx]] = shading_values[sort_idx]

    specular_values = specular[orig_x, orig_y]
    sort_idx = np.argsort(specular_values[:, 1], axis=0)  # from dim to bright
    specular_sp[warped_x[sort_idx], warped_y[sort_idx]] = specular_values[sort_idx]
    
    # if verbose:
    #     print(f"Fast version: time used: {time.time() - start_time:.6f} sec(s)")
        
    # adjust max pooling kernel size according to input and output size
    ksize = int((out_hw + 255) / 256) * 2 + 1 if out_hw > 256 else 1  # 257 ~ 512 -> 3; 513 ~ 768 -> 5; 769 ~ 1024 -> 7
    in_hw = mask.shape[0]
    while in_hw <= 256:
        ksize = ksize * 2 - 1
        in_hw *= 2
    if ksize > 1:
        max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(ksize, ksize))
        shading_sp = cv2.dilate(shading_sp, max_kernel)
        specular_sp = cv2.dilate(specular_sp, max_kernel)
        mask_sp = cv2.dilate(mask_sp, max_kernel)
        if len(mask_sp.shape) == 2:
            mask_sp = mask_sp[:, :, None]
    
    return {
        "specular": specular_sp,
        "shading": shading_sp,
        "mask": mask_sp,
    }


def generate_sphere_mask_and_normal(out_hw=512, print_to_file=False):
    """
    Generate sphere mask and normal of shape (out_hw, out_hw, c).
    """
    # generate row and column index
    idx = np.arange(out_hw)
    idx_row = np.repeat(idx[:, None], repeats=out_hw, axis=1)
    idx_col = np.repeat(idx[None, :], repeats=out_hw, axis=0)
    # generate each component of normal
    normal_x = (idx_col - out_hw / 2 + 0.5) / (out_hw / 2)
    normal_y = -(idx_row - out_hw / 2 + 0.5) / (out_hw / 2)
    normal_z = (np.maximum((1 - normal_x ** 2 - normal_y ** 2), 0)) ** 0.5
    # generate mask and normal
    mask_sp = (normal_x ** 2 + normal_y ** 2 <= 1)[:, :, None].astype(np.float32)
    normal_sp = np.stack([normal_x, normal_y, normal_z], axis=2) * mask_sp
    # visualize
    if print_to_file:
        write_image(os.path.join(img_dir, "mask_sp.png"), mask_sp)
        write_image(os.path.join(img_dir, "normal_sp.png"), (normal_sp + 1) / 2 * mask_sp, depth=16)
    return mask_sp, normal_sp


def xyz2polar(x, y, z):
    """
    Transform a unit vector from (x, y, z) to (φ, θ).
    Args:
        x: positive is right
        y: positive is up
        z: positive is outward
    Returns:
        theta: latitude
        phi: longitude
    """
    theta = np.arcsin(y)
    phi = np.arctan2(x, z)
    return theta, phi


def warp_rect2sp(
        envmap_rect: np.ndarray,
        mask_sp: np.ndarray = None,
        normal_sp: np.ndarray = None,
        out_hw: int = 512,
        verbose: bool = False):
    """Warp a rectangular envmap into a spherical envmap.
    
    Args:
        envmap_rect: [h, w, 3], the rectangular envmap to be warped to sphere.
        mask_sp: [h, w, 1], the sphere mask. Can be generated using 'generate_sphere_mask_and_normal'.
        normal_sp: [h, w, 3], the sphere surface normal. Can be generated using 'generate_sphere_mask_and_normal'.
        out_hw: the width and height of output image.
        verbose: if True, print time used.
        
    Returns:
        Spherical envmap
    """
    
    if mask_sp is None and normal_sp is None:
        mask_sp, normal_sp = generate_sphere_mask_and_normal(out_hw=out_hw, print_to_file=False)
        
    envmap_sp = np.zeros(shape=(out_hw, out_hw, 3), dtype=np.float32)
    if envmap_rect.shape[0] > out_hw:
        envmap_rect = cv2.resize(envmap_rect, dsize=(out_hw * 2, out_hw),
                                 interpolation=cv2.INTER_AREA)

    in_h, in_w, _ = envmap_rect.shape
    envmap_rect = np.concatenate([envmap_rect, envmap_rect[-1:, :]], axis=0)
    envmap_rect = np.concatenate([envmap_rect, envmap_rect[:, :1]], axis=1)
    view_vec = np.array([0, 0, 1], dtype=np.float32)  # orthographic camera
    
    # ========== Fast Version ========== (~0.07 sec for 1 image)
    start_time = time.time()
    # get pixel coordinates
    sp_x, sp_y = np.where(mask_sp[:, :, 0] > 0.5)
    sp_normal = normal_sp[sp_x, sp_y]
    # compute reflect vector, transform to polar coord.
    refl_vec = sp_normal * sp_normal[:, 2:] * 2.0 - view_vec
    theta, phi = xyz2polar(refl_vec[:, 0], refl_vec[:, 1], refl_vec[:, 2])
    # find position on rectangular envmap
    rect_x = in_h / 2 - theta / (np.pi / 2) * in_h / 2
    rect_y = in_w / 2 - phi / np.pi * in_w / 2
    # # nearest neighbor
    # envmap_sp[warped_x, warped_y] = envmap_rect[orig_x.astype(int), orig_y.astype(int)]
    # do bilinear interpolation
    x_down = rect_x.astype(int)
    x_up = x_down + 1
    y_down = rect_y.astype(int)
    y_up = y_down + 1
    dd = envmap_rect[x_down, y_down]
    du = envmap_rect[x_down, y_up]
    ud = envmap_rect[x_up, y_down]
    uu = envmap_rect[x_up, y_up]
    ddw = (x_up - rect_x) * (y_up - rect_y)
    duw = (x_up - rect_x) * (rect_y - y_down)
    udw = (rect_x - x_down) * (y_up - rect_y)
    uuw = (rect_x - x_down) * (rect_y - y_down)
    envmap_sp[sp_x, sp_y] = ddw[..., None] * dd + duw[..., None] * du \
                                    + udw[..., None] * ud + uuw[..., None] * uu
    if verbose:
        print(f"Fast version: time used: {time.time() - start_time:.6f} sec(s)")
            
    return envmap_sp #, mask_sp, normal_sp

def warp_rect_to_sphere(envmap_rect: np.ndarray,
                        mask_sp: np.ndarray = None,
                        normal_sp: np.ndarray = None,
                        out_hw: int = 512,
                        verbose: bool = False):
    """Warp a rectangular envmap into a spherical envmap.
    
    Args:
        envmap_rect: [h, w, 3], the rectangular envmap to be warped to sphere.
        mask_sp: [h, w, 1], the sphere mask. Can be generated using 'generate_sphere_mask_and_normal'.
        normal_sp: [h, w, 3], the sphere surface normal. Can be generated using 'generate_sphere_mask_and_normal'.
        out_hw: the width and height of output image.
        verbose: if True, print time used.
        
    Returns:
        Spherical envmap; spherical mask; spherical normal.
    """
    if mask_sp is None and normal_sp is None:
        mask_sp, normal_sp = generate_sphere_mask_and_normal(out_hw=out_hw, print_to_file=False)
    
    envmap_sp = np.zeros(shape=(out_hw, out_hw, 3), dtype=np.float32)


    in_h, in_w, _ = envmap_rect.shape
    envmap_rect = np.concatenate([envmap_rect, envmap_rect[-1:, :]], axis=0)
    envmap_rect = np.concatenate([envmap_rect, envmap_rect[:, :1]], axis=1)
    view_vec = np.array([0, 0, 1], dtype=np.float32)  # orthographic camera
    
    # ========== Fast Version ========== (~0.07 sec for 1 image)
    start_time = time.time()
    # get pixel coordinates
    warped_x, warped_y = np.where(mask_sp[:, :, 0] > 0.5)
    warped_normal = normal_sp[warped_x, warped_y]
    # compute reflect vector, transform to polar coord.
    refl = warped_normal * warped_normal[:, 2:] * 2.0 - view_vec
    theta, phi = xyz2polar(refl[:, 0], refl[:, 1], refl[:, 2])
    # find position on rectangular envmap
    orig_x = in_h / 2 - theta / (np.pi / 2) * in_h / 2
    orig_y = in_w / 2 - phi / np.pi * in_w / 2
    # # nearest neighbor
    # envmap_sp[warped_x, warped_y] = envmap_rect[orig_x.astype(int), orig_y.astype(int)]
    # do bilinear interpolation
    x_down = orig_x.astype(int)
    x_up = x_down + 1
    y_down = orig_y.astype(int)
    y_up = y_down + 1
    dd = envmap_rect[x_down, y_down]
    du = envmap_rect[x_down, y_up]
    ud = envmap_rect[x_up, y_down]
    uu = envmap_rect[x_up, y_up]
    ddw = (x_up - orig_x) * (y_up - orig_y)
    duw = (x_up - orig_x) * (orig_y - y_down)
    udw = (orig_x - x_down) * (y_up - orig_y)
    uuw = (orig_x - x_down) * (orig_y - y_down)
    envmap_sp[warped_x, warped_y] = ddw[..., None] * dd + duw[..., None] * du \
                                    + udw[..., None] * ud + uuw[..., None] * uu
    if verbose:
        print(f"Fast version: time used: {time.time() - start_time:.6f} sec(s)")
            
    return envmap_sp, mask_sp, normal_sp
            

def polar2xyz(theta, phi):
    """
    Transform polar coord (φ, θ) to a unit vector (x, y, z).
    Args:
        theta: latitude
        phi: longitude
    Returns:
        x: positive is right
        y: positive is up
        z: positive is outward
    """
    y = np.sin(theta)
    x = np.cos(theta) * np.sin(phi)
    z = np.cos(theta) * np.cos(phi)
    return x, y, z



def warp_sp2rect(
        envmap_sp: np.ndarray,
        mask_sp: np.ndarray = None,
        out_h: int = 512,
        verbose: bool = False):
    """Warp a spherical envmap into a rectangular envmap.
    
    Args:
        envmap_sp: [h, w, 3], the spherical envmap to be warped. h should be equal to w.
        mask_sp: [h, w, 1], the sphere mask.
        out_h: the height of output image. output width = height * 2.
        verbose: if True, print time used.
        
    Returns:
        Rect envmap.
    """
    # st()

    if mask_sp is None :
        mask_sp, _ = generate_sphere_mask_and_normal(out_hw=envmap_sp.shape[0], print_to_file=False)
    
    envmap_rect = np.zeros(shape=(out_h, out_h * 2, 3), dtype=np.float32)
    mask_rect = np.ones(shape=(out_h, out_h * 2), dtype=np.float32)
    out_w = out_h * 2
    if envmap_sp.shape[0] > out_h:
        envmap_sp = cv2.resize(envmap_sp, dsize=(out_h, out_h),
                               interpolation=cv2.INTER_AREA)
        mask_sp = cv2.resize(mask_sp, dsize=(out_h, out_h),
                               interpolation=cv2.INTER_NEAREST)
    
    assert envmap_sp.shape[0] == envmap_sp.shape[1]
    in_hw = envmap_sp.shape[0]
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    mask_sp = cv2.erode(mask_sp, kernel=kernel)[:, :, None]
    envmap_sp_dilate = cv2.dilate(envmap_sp, kernel=kernel)
    # print(f"envmap_sp_dilate.shape: {envmap_sp_dilate.shape}")
    # print(f"mask_sp.shape: {mask_sp.shape}")
    # print(f"envmap_sp.shape: {envmap_sp.shape}")

    envmap_sp = envmap_sp * mask_sp + envmap_sp_dilate * (1 - mask_sp)
    # write_image(os.path.join(img_dir, "envmap_sp_dilate.hdr"), envmap_sp)
    envmap_sp = np.concatenate([envmap_sp, envmap_sp[-1:, :]], axis=0)
    envmap_sp = np.concatenate([envmap_sp, envmap_sp[:, -1:]], axis=1)
    view_vec = np.array([0, 0, 1], dtype=np.float32)  # orthographic camera

    start_time = time.time()
    rect_x, rect_y = np.where(mask_rect > 0.5)  # flatten
    theta = (out_h / 2 - rect_x) / out_h * 2 * np.pi / 2
    phi = (rect_y - out_w / 2) / out_w * 2 * np.pi
    # compute half vector
    half = np.zeros(shape=(rect_x.shape[0], 3), dtype=np.float32)
    half[..., 0], half[..., 1], half[..., 2] = polar2xyz(theta, phi)
    half = (half + view_vec) / 2
    half = half / (np.sqrt(np.sum(half ** 2, axis=1, keepdims=True)) + 1e-12)
    # at_border = (half[..., 0] ** 2 + half[..., 1] ** 2) > 1 - 1 / in_hw
    # half[at_border] = half[at_border] * 0.995
    # find position on spherical envmap
    sp_x = in_hw / 2 - half[..., 1] * in_hw / 2
    sp_y = in_hw / 2 - half[..., 0] * in_hw / 2
    # do bilinear interpolation
    x_down = sp_x.astype(int)
    x_up = x_down + 1
    y_down = sp_y.astype(int)
    y_up = y_down + 1
    dd = envmap_sp[x_down, y_down]
    du = envmap_sp[x_down, y_up]
    ud = envmap_sp[x_up, y_down]
    uu = envmap_sp[x_up, y_up]
    ddw = (x_up - sp_x) * (y_up - sp_y)
    duw = (x_up - sp_x) * (sp_y - y_down)
    udw = (sp_x - x_down) * (y_up - sp_y)
    uuw = (sp_x - x_down) * (sp_y - y_down)
    envmap_rect[rect_x, rect_y] = ddw[..., None] * dd + duw[..., None] * du \
                                  + udw[..., None] * ud + uuw[..., None] * uu
    if verbose:
        print(f"Fast version: time used: {time.time() - start_time:.6f} sec(s)")

    return envmap_rect



if __name__ == "__main__":
    
    # img_dir = r"D:\dataset\Experimental\4"
    
    # normal = read_image(os.path.join(img_dir, "normal_gt.exr"))
    # if np.min(normal) > -0.1:  # in [0, 1]
    #     normal = normal * 2 - 1  # transform to [-1, 1]
    # normal = normal / np.sqrt((np.sum(normal ** 2, axis=2, keepdims=True) + 1e-6))
    # mask = read_image(os.path.join(img_dir, "mask.png"))
    # shading = read_image(os.path.join(img_dir, "shading_gt.hdr"))
    # specular = read_image(os.path.join(img_dir, "specular_gt.hdr"))
    
    # if len(mask.shape) == 2:
    #     mask = mask[:, :, None]

    # write_image(os.path.join(img_dir, "normal_gt.png"), (normal + 1) / 2 * mask)

    # warped = warp_face_to_sphere({"normal": normal,
    #                               "mask": mask,
    #                               "shading": shading,
    #                               "specular": specular},
    #                              out_hw=512,
    #                              verbose=True)

    # write_image(os.path.join(img_dir, "specular_sp_gt.hdr"), warped["specular"])
    # write_image(os.path.join(img_dir, "shading_sp_gt.hdr"), warped["shading"])
    # write_image(os.path.join(img_dir, "mask_sp_gt.png"), warped["mask"])

    #
    mask_sp, normal_sp = generate_sphere_mask_and_normal(out_hw=128)
    write_image('./visualize_paper/normal_sp.png', normal_sp)
    #
    # envmap = read_image(os.path.join(img_dir, "envmap_gt.hdr"))
    # envmap_sp = warp_rect_to_sphere(envmap, mask_sp, normal_sp)
    # write_image(os.path.join(img_dir, "envmap_sp_gt.hdr"), envmap_sp)