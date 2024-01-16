"""
Pre-processing the faces: detect faces, crop, and generate face region masks.
The face semantic segmentation model and the corresponding codes are from:
    https://github.com/royorel/FFHQ-Aging-Dataset
"""
from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import torch
from numpy import ndarray
from torchvision import transforms
from tqdm import tqdm

from models import deeplab
from utils.general import read_image, read_mask, write_image

CLASSES = [
    "background",
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]
FACE_REGION_CLASSES = ["skin", "nose", "eye_g", "l_brow", "r_brow", "u_lip", "l_lip"]


def detect_and_crop(
    image: np.ndarray, crop_size: int = 512, oversizing: float = 1.1,
) -> Union[Tuple[None, None], Tuple[Any, ndarray]]:
    """Crop the image to a square image with crop_size containing the face.

    Crop the image to a square image with <crop_size> containing the face.
    Use mediapipe to detect the face, then compute a bounding box with some
    margins around the face.

    Args:
        image (np.ndarray): the image to be cropped.
        crop_size (int): the size of the cropped image.
        oversizing (float): the oversizing factor of the bounding box.
    """
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        # detect the face
        detect_res = face_detection.process((image * 255).astype(np.uint8))
        if detect_res.detections is None:
            return None, None
        # get the bounding box of the face
        bbox = detect_res.detections[0].location_data.relative_bounding_box
        bbox = np.array([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
        bbox = bbox * np.array(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        )
        box_center = bbox[:2] + bbox[2:] / 2
        # add some margins and make the bbox square
        half_size = max(bbox[2], bbox[3]) / 2 * oversizing
        # make sure the bbox is within the image
        half_size = min(half_size, min(box_center[0], image.shape[1] - box_center[0]))
        half_size = min(half_size, min(box_center[1], image.shape[0] - box_center[1]))
        bbox = np.array(
            [
                box_center[0] - half_size,
                box_center[1] - half_size,
                half_size * 2,
                half_size * 2,
            ]
        )
        bbox = bbox.astype(int)
        bbox[3] += bbox[1]
        bbox[2] += bbox[0]
        # crop the image
        crop_img = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        # resize the image
        crop_img = cv2.resize(crop_img, (crop_size, crop_size))
        return crop_img, bbox


def segment_image(
    image: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    include_eye: bool = False,
    include_mouth: bool = False,
) -> np.ndarray:
    """Segment the image and get the face region mask using segmentation model.

    Args:
        image (np.ndarray): the image to be segmented.
        model (torch.nn.Module): the segmentation model.
        transform (torchvision.transforms.Compose): the data transform.
        include_eye (bool): whether to include the eye region.
        include_mouth (bool): whether to include the mouth region.

    Returns:
        np.ndarray: (h, w, 1) the mask of the face region.
    """
    image = transform(image)
    _, h, w = image.shape
    image = image.cuda()
    pred = model(image.unsqueeze(0))
    _, pred = torch.max(pred, 1)
    mask = torch.zeros_like(pred, dtype=torch.uint8)
    face_region_indices = [CLASSES.index(c) for c in FACE_REGION_CLASSES]
    if include_eye:
        face_region_indices += [CLASSES.index("l_eye"), CLASSES.index("r_eye")]
    if include_mouth:
        face_region_indices += [CLASSES.index("mouth")]
    for cls in face_region_indices:
        mask[pred == cls] = 255
    mask = mask.data.cpu().numpy().squeeze().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(float) / 255.0
    mask = mask[:, :, None]
    return mask


def get_segment_model():
    """Prepare the segmentation model and data transform."""
    assert torch.cuda.is_available()
    model_fname = ckpt_dir / "deeplab_model.pth"
    model = getattr(deeplab, "resnet101")(
        pretrained=False,
        num_classes=len(CLASSES),
        num_groups=32,
        weight_std=True,
        beta=False,
    )
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname)

    state_dict = {
        k[7:]: v for k, v in checkpoint["state_dict"].items() if "tracked" not in k
    }
    model.load_state_dict(state_dict)

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return model, data_transforms


if __name__ == "__main__":
    img_dir = Path("./images")
    ckpt_dir = Path("./model_files")
    face_dir = img_dir / "face"
    mask_dir = img_dir / "mask"
    pre_proc_dir = img_dir / "pre_processed"
    pre_proc_dir.mkdir(exist_ok=True)

    model, transform = get_segment_model()

    # find all image files with suffix .png, .jpg, .jpeg
    face_paths = list(face_dir.glob("*.*"))
    face_paths = [p for p in face_paths if p.suffix in [".png", ".jpg", ".jpeg"]]
    print(f"Found {len(face_paths)} faces in {face_dir.absolute()}")

    # pre-process faces: crop and segment
    for face_path in tqdm(face_paths, desc="Pre-processing faces"):
        # if masks are given, use them and copy the original face
        if (mask_dir / (face_path.stem + ".png")).exists():
            mask_img = read_mask(mask_dir / (face_path.stem + ".png"))
            face_img = read_image(face_path)
            write_image(pre_proc_dir / (face_path.stem + "_mask.png"), mask_img)
            write_image(pre_proc_dir / (face_path.stem + ".png"), face_img)
            write_image(
                pre_proc_dir / (face_path.stem + "_masked.png"), face_img * mask_img,
            )
        else:
            # crop and segment
            face_cropped, _ = detect_and_crop(read_image(face_path), crop_size=512)
            mask_img = segment_image(
                face_cropped, model, transform, include_eye=False, include_mouth=False
            )
            # erode without changing eye
            mask_orig = (mask_img * 255).astype(np.uint8)
            mask_eye = mask_orig.copy()
            cv2.floodFill(
                mask_eye, mask=None, seedPoint=(0, 0), newVal=(255,),
            )
            mask_eye = 255 - mask_eye
            mask_filled = mask_orig + mask_eye
            face_kernel = cv2.getStructuringElement(
                shape=cv2.MORPH_ELLIPSE, ksize=(25, 25)
            )
            mask_eroded = cv2.erode(mask_filled, kernel=face_kernel)[:, :, None]
            eye_kernel = cv2.getStructuringElement(
                shape=cv2.MORPH_ELLIPSE, ksize=(10, 10)
            )
            mask_eye = cv2.dilate(mask_eye, kernel=eye_kernel)[:, :, None]
            mask_img = (mask_eroded - mask_eye).astype(float) / 255.0
            write_image(pre_proc_dir / (face_path.stem + ".png"), face_cropped)
            write_image(pre_proc_dir / (face_path.stem + "_mask.png"), mask_img)
            write_image(
                pre_proc_dir / (face_path.stem + "_masked.png"), face_cropped * mask_img
            )
