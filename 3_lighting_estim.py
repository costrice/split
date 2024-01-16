from pathlib import Path

import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models.lighting_networks import LightingEstimator
from utils.general import (
    linrgb2srgb,
    read_image,
    read_mask,
    tensor2ndarray,
    write_image,
)
from utils.geotransform import generate_sphere_mask_and_normal
from utils.ibrender import IBRenderer

if __name__ == "__main__":
    img_dir = Path("./images")
    ckpt_dir = Path("./model_files")
    pre_proc_dir = img_dir / "pre_processed"
    intrinsic_dir = img_dir / "intrinsic"
    lighting_dir = img_dir / "lighting"
    lighting_dir.mkdir(exist_ok=True)
    lighting_dirs = {}

    _, normal_sp = generate_sphere_mask_and_normal(out_hw=64)
    normal_sp = TF.to_tensor(normal_sp).float().cuda().unsqueeze(0)

    networks = {}
    for scene in ["indoor", "outdoor"]:
        networks[scene] = LightingEstimator(
            ckpt_path=ckpt_dir / f"lighting_estim_{scene}.pth", device="cuda:0",
        )
        networks[scene].eval()
        lighting_dirs[scene] = lighting_dir / scene
        lighting_dirs[scene].mkdir(exist_ok=True)

    # find all image files in pre_proc_dir
    face_paths = list(pre_proc_dir.glob("*.*"))
    face_paths = [
        p
        for p in face_paths
        if p.suffix == ".png" and not p.stem.endswith(("_mask", "_masked"))
    ]

    ibrender = IBRenderer(ridx=4, device="cuda:0", bases_folder=r"model_files/")

    for face_path in tqdm(face_paths, desc="Estimating lighting"):
        # read distributed shading and specular
        face_name = face_path.stem
        specular_sp = read_image(intrinsic_dir / (face_name + "_distS.hdr"))
        shading_sp = read_image(intrinsic_dir / (face_name + "_distD.hdr"))
        mask_sp = read_mask(intrinsic_dir / (face_name + "_distM.png"))

        specular_sp = TF.to_tensor(specular_sp).float().cuda().unsqueeze(0)
        shading_sp = TF.to_tensor(shading_sp).float().cuda().unsqueeze(0)
        mask_sp = TF.to_tensor(mask_sp).float().cuda().unsqueeze(0)

        for scene in ["indoor", "outdoor"]:
            scene_abbr = scene[0].upper()
            net = networks[scene]
            save_dir = lighting_dirs[scene]

            # infer lighting
            untex_hdr, tex_ldr, tex_hdr = net(
                {
                    "specular": specular_sp,
                    "shading": shading_sp,
                    "mask": mask_sp,
                    "normal": normal_sp,
                }
            )

            # save results
            pred_ibr = ibrender.render_using_mirror_sphere(tex_hdr)

            pred_mirror = tensor2ndarray(tex_hdr[0])
            pred_spec = tensor2ndarray(pred_ibr["specular"][0])
            pred_diff = tensor2ndarray(pred_ibr["diffuse"][0])
            multiplier = 0.3 / np.mean(pred_diff)
            pred_diff *= multiplier
            pred_spec *= multiplier

            write_image(save_dir / (face_name + f"_pred.hdr"), pred_mirror)
            write_image(
                save_dir / (face_name + f"_ibr_diff.png"), linrgb2srgb(pred_diff),
            )
            write_image(
                save_dir / (face_name + f"_ibr_spec.png"), linrgb2srgb(pred_spec),
            )
