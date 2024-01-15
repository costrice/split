"""Move files in """
import shutil
from pathlib import Path

import numpy as np
import os

import cv2
# from codes.lighting_estimation.codes.utils import general

src_dir = Path(r"F:\Datasets\Face2LightCollection\LightTestIntrinsicResults")
dst_dir = Path(r"F:\Datasets\SPLiT-Release\SPLiT Face&Lighting")
# dst_dir_face = dst_dir / 'face'
# dst_dir_lighting = dst_dir / 'lighting'
# dst_dir_mask = dst_dir / 'mask'
# remove existing files
# if dst_dir.exists():
#     shutil.rmtree(dst_dir)
# for dst_d in [dst_dir_face, dst_dir_lighting, dst_dir_mask]:
#     dst_d.mkdir(parents=True, exist_ok=True)

# dst_dir = dst_dir / scene_category

data_path_list = Path(
    r"E:\Codes\SPLiT\codes\lighting_estimation\test_list\laval_cvpr_Syn+Real_v2.csv"
)
out_prefix = "O-"

# data_path_list = Path(
#     r"E:\Codes\SPLiT\codes\lighting_estimation\test_list\indoor.csv"
# )
# out_prefix = 'I-'

def srgb2linrgb(color_srgb):
    """
    Transform a image in [0, 1] from sRGB to linear RGB space.
    """
    big = color_srgb > 0.0404482362771082
    color_linrgb = big * (((color_srgb + 0.055) / 1.055) ** 2.4) + (~big) * (
        color_srgb / 12.92
    )
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
            print(f"ValueError: Failed to read image {img_path}")
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
    
if __name__ == '__main__':

    with open(data_path_list, "r") as f:
        data_path_list = f.readlines()
        data_path_list = [x.strip() for x in data_path_list]
        for data_path in data_path_list:
            if not data_path.startswith("/userhome"):
                continue
            # example path: /userhome/feifan/dataset/Face2LightReal/outdoor-1/decom_by_I2AN2DS_Syn+Realv2/DSC07271
            # get the names starting from Face2LightReal
            data_path = Path(data_path).parts[-4:]
            if data_path[0] != "Face2LightReal":
                continue
            print("Processing", data_path)
            scene_dir = src_dir / data_path[1]
            face_dir = scene_dir / "face_cropped"
            lighting_dir = scene_dir / "envmap_proc"
            mask_dir = scene_dir / "face_parsing_mask"
            # copy files
            src_name = data_path[-1]
            dst_name = src_name
            if dst_name.startswith("CI_"):
                dst_name = out_prefix + f"{9000+int(dst_name[3:]):04d}"
            else:  # begins with 'DSC'
                dst_name = out_prefix + f"{int(dst_name[3:]):04d}"
                
            # read face image and convert to linear space
            face_img = read_image(str(face_dir / (src_name + ".png")))
            face_img = srgb2linrgb(face_img)
            write_image(str(dst_dir / (dst_name + "-lin.png")), face_img)
    
            # shutil.copy(
            #     face_dir / (src_name + ".png"), dst_dir / "face" / (dst_name + ".png")
            # )
            shutil.copy(
                lighting_dir / "envmap.hdr", dst_dir / (dst_name + ".hdr")
            )
            shutil.copy(
                mask_dir / (src_name + ".png"), dst_dir / (dst_name + "-mask.png")
            )
