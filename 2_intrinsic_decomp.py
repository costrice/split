from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models import intrinsic_networks
from utils.general import (
    linrgb2srgb,
    read_image,
    read_mask,
    srgb2linrgb,
    tensor2ndarray,
    write_image,
)
from utils.geotransform import distribute_face_to_sphere

if __name__ == "__main__":
    img_dir = Path("./images")
    ckpt_dir = Path("./model_files")
    pre_proc_dir = img_dir / "pre_processed"
    intrinsic_dir = img_dir / "intrinsic"
    intrinsic_dir.mkdir(exist_ok=True)

    # create model and load weights
    net_f_an = intrinsic_networks.I2ANNet(base_filters=48)
    net_f_an.load_state_dict(
        torch.load(ckpt_dir / "intrinsic_decom_f_AN.pth", map_location="cpu")["model"]
    )
    net_f_ds = intrinsic_networks.IAN2DSNet(base_filters=48)
    net_f_ds.load_state_dict(
        torch.load(ckpt_dir / "intrinsic_decom_f_DS.pth", map_location="cpu")["model"]
    )
    net_cascade = intrinsic_networks.CascadeNetwork(
        model0=net_f_an, dev0=0, model1=net_f_ds, dev1=0
    )
    net_cascade.eval()

    # find all image files in pre_proc_dir
    face_paths = list(pre_proc_dir.glob("*.*"))
    face_paths = [
        p
        for p in face_paths
        if p.suffix == ".png" and not p.stem.endswith(("_mask", "_masked"))
    ]

    for face_path in tqdm(face_paths, desc="Decomposing face intrinsics"):
        face_name = face_path.stem
        mask_path = pre_proc_dir / (face_name + "_mask.png")
        face = read_image(face_path)
        face = srgb2linrgb(face)
        mask = read_mask(mask_path)

        face = TF.to_tensor(face).float().cuda().unsqueeze(0)
        mask = TF.to_tensor(mask).float().cuda().unsqueeze(0)

        # construct input
        net_input = {"face": face, "mask": mask}

        # evaluate and save results
        with torch.no_grad():
            pred = net_cascade(net_input)

        normal = tensor2ndarray(pred["normal"][0])
        albedo = tensor2ndarray(pred["albedo"][0])
        shading = tensor2ndarray(pred["shading"][0])
        specular = tensor2ndarray(pred["specular"][0])
        face_recon = tensor2ndarray(pred["face"][0])
        face = tensor2ndarray(face[0])
        mask = tensor2ndarray(mask[0])
        face_recon = face_recon * mask + face * (1 - mask)

        # save results
        write_image(
            intrinsic_dir / (face_name + "_intrA.png"), linrgb2srgb(albedo),
        )
        write_image(
            intrinsic_dir / (face_name + "_intrN.png"),
            (normal + 1) / 2 * mask,
            depth=16,
        )
        write_image(
            intrinsic_dir / (face_name + "_intrD.hdr"), shading,
        )
        write_image(
            intrinsic_dir / (face_name + "_intrS.hdr"), specular,
        )
        write_image(
            intrinsic_dir / (face_name + "_intrRecon.png"), linrgb2srgb(face_recon),
        )

        # distribute according to surface normal
        distributed = distribute_face_to_sphere(
            {"normal": normal, "mask": mask, "shading": shading, "specular": specular,},
            out_hw=64,
            choice="max",
        )
        write_image(
            intrinsic_dir / (face_name + "_distD.hdr"), distributed["shading"],
        )
        write_image(
            intrinsic_dir / (face_name + "_distS.hdr"), distributed["specular"],
        )
        write_image(
            intrinsic_dir / (face_name + "_distM.png"), distributed["mask"],
        )
