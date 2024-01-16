# -*- coding: utf-8 -*-
"""
General utility functions.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from dateutil import tz


def current_time() -> str:
    """Return current time as a string."""
    return datetime.now().astimezone(tz.gettz("UTC+8")).strftime("%m%d%H%M")


def tensor2ndarray(img) -> np.ndarray:
    """Convert a tensor to a numpy ndarray.
    The tensor must not contain batch dimension.
    """
    if len(img.size()) == 4:
        print(
            "Warning: tensor2ndarray() received a tensor with batch "
            "dimension, which will be squeezed."
        )
        img = img[0]
    return img.cpu().detach().numpy().transpose(1, 2, 0)


def linrgb2srgb(color_linrgb):
    """Transform an image in [0, 1] from linear sRGB to sRGB space."""
    big = color_linrgb > 0.0031308
    color_srgb = color_linrgb * 0
    color_srgb[big] = 1.055 * (color_linrgb[big] ** (1 / 2.4)) - 0.055
    color_srgb[~big] = color_linrgb[~big] * 12.92
    # color_srgb = color_linrgb ** (1 / 2.2)
    return color_srgb


def srgb2linrgb(color_srgb):
    """Transform an image in [0, 1] from sRGB to linear sRGB space."""
    big = color_srgb > 0.0404482362771082
    color_linrgb = color_srgb * 0
    color_linrgb[big] = ((color_srgb[big] + 0.055) / 1.055) ** 2.4
    color_linrgb[~big] = color_srgb[~big] / 12.92
    # color_linrgb = color_srgb ** 2.2
    return color_linrgb


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.uint8. If float, clip to [0, 1] first.
    """
    if img.dtype == bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 255).astype(np.uint8)
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, 0, 1) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def to_uint16(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.uint16. If float, clip to [0, 1] first.
    """
    if img.dtype == bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 255
    if img.dtype == np.uint16:
        return img
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, 0, 1) * 65535).astype(np.uint16)
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert an image into np.float32
    """
    if img.dtype == bool:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535
    if img.dtype == np.float32:
        return img
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def read_image(img_path: Union[str, Path]) -> np.ndarray:
    """
    read an image and convert to np.float32 (range in [0, 1] if image is LDR).

    Args:
        img_path: path to the image.

    Returns:
        ndarray, dtype=np.float32
    """
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if os.path.exists(img_path):
        img = cv2.imread(img_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"File {img_path} cannot be read.")
        if len(img.shape) == 3:  # BGR
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = to_float32(img)
        return img
    else:
        raise FileNotFoundError(f"File {img_path} not found.")


def write_image(output_path: Union[str, Path], img: np.ndarray, depth: int = 8):
    """
    Write a ndarray <img> into <output_path>, LDR or HDR according to extension
    specified in <output_path>.

    Args:
        output_path: the save path.
        img: the image to be saved.
        depth: bit depth, can be 8 or 16.
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)
    if img.dtype in [np.float64, bool]:  # cv2 do not support float64?
        img = img.astype(np.float32)
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
    if output_path.endswith((".hdr", ".exr")):
        cv2.imwrite(output_path, to_float32(img))
    elif output_path.endswith((".png")):
        if depth == 8:
            cv2.imwrite(output_path, to_uint8(img))
        elif depth == 16:
            cv2.imwrite(output_path, to_uint16(img))
        else:
            raise ValueError(f"Unexpected depth {depth}")
    elif output_path.endswith((".jpg")):
        if depth == 8:
            cv2.imwrite(output_path, to_uint8(img))
        else:
            raise ValueError(f"Unexpected depth {depth}")
    else:
        raise ValueError(f"Unexpected file extension in {output_path}")


def read_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """Read a mask image and make its shape (h, w, 1) and its type float."""
    if isinstance(mask_path, Path):
        mask_path = str(mask_path)
    mask_img = read_image(mask_path)
    if len(mask_img.shape) == 2:
        mask_img = mask_img[:, :, None]
    else:
        mask_img = mask_img[:, :, :1]
    mask_img = (mask_img > 0.5).astype(float)
    return mask_img


if __name__ == "__main__":
    example_dir = Path(r"./images/examples")
    example_dir.mkdir(exist_ok=True)

    random_srgb = np.random.uniform(low=0, high=1, size=(400, 400, 3))
    random_linrgb = srgb2linrgb(random_srgb)
    random_srgb_recon = linrgb2srgb(random_linrgb)
    write_image(example_dir / "random_srgb.png", random_srgb)
    write_image(example_dir / "random_srgb_recon.png", random_srgb_recon)
    write_image(example_dir / "random_linrgb.png", random_linrgb)
