"""
This code implements image-based rendering, which is differentiable.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF


class IBRenderer:
    def __init__(
        self, ridx: int = None, device: str = "cpu", bases_folder: str = None,
    ):
        self.bases_folder = Path(bases_folder)
        self.envmap_size = 64
        self.device = device
        self.load_bases(ridx)

    def get_diff_name(self, envmap_size: int):
        """A rememberer of directory name for diffuse BRDF."""
        return f"diff_s{envmap_size}"

    def get_spec_name(self, envmap_size: int, roughness_idx: int):
        """A rememberer of directory name for glossy BRDF."""
        return f"spec_r{roughness_idx:02d}_s{envmap_size}"

    def get_ms_name(self, envmap_size: int):
        """A rememberer of directory name for glossy BRDF."""
        return f"google_ms_s{envmap_size}"

    def load_bases(self, ridx: int):
        """Loads bases from .npy files.

        Args:
            ridx: index of specular BRDF.
        """
        self.transform_mat_file_diff = self.bases_folder / (
            self.get_diff_name(self.envmap_size) + ".npy"
        )
        transform_mat_diff = np.load(str(self.transform_mat_file_diff))
        self.transform_mat_diff = TF.to_tensor(transform_mat_diff)[0]

        self.transform_mat_file_spec = self.bases_folder / (
            self.get_spec_name(self.envmap_size, ridx) + ".npy"
        )

        transform_mat_spec = np.load(str(self.transform_mat_file_spec))
        self.transform_mat_spec = TF.to_tensor(transform_mat_spec)[0]

        if self.device != "cpu":
            self.transform_mat_diff = self.transform_mat_diff.to(self.device)
            self.transform_mat_spec = self.transform_mat_spec.to(self.device)

    def render_using_mirror_sphere(
        self,
        sphere_mirror: torch.Tensor,
        transform_mat_diff: torch.Tensor = None,
        transform_mat_spec: torch.Tensor = None,
    ):
        """Renders diffuse and specular ball under the input mirror ball as environment map.

        Args:
            sphere_mirror: the input mirror ball under which other two ball are rendered.
            ridx: Roughness level, a integer in [0, len(config.roughness_settings) ).
            transform_mat_diff: preloaded transform matrix for diffuse ball. If `None`, reload it.
            transform_mat_spec: preloaded transform matrix for glossy ball. If `None`, reload it.

        Returns:
            Dict[str, torch.Tensor]: a dict, in which "diffuse" contains rendered diffuse ball,
                and "specular" contains glossy ball.
        """

        def helper(transform_mat_file, transform_mat=None):
            if transform_mat is None:
                transform_mat = np.load(transform_mat_file)
                transform_mat = TF.to_tensor(transform_mat)[0]

            transform_mat = transform_mat.to(sphere_mirror)
            sphere_ibr = (
                sphere_mirror.reshape(in_b, in_c, -1) @ transform_mat
            ).reshape(in_b, in_c, self.envmap_size, self.envmap_size)
            sphere_ibr /= 256  # calibrated intensity correction coefficient
            return sphere_ibr

        if transform_mat_diff is None:
            transform_mat_diff = self.transform_mat_diff
        if transform_mat_spec is None:
            transform_mat_spec = self.transform_mat_spec

        rendered = {}
        in_b, in_c, in_h, in_w = sphere_mirror.shape
        assert in_h == self.envmap_size and in_w == self.envmap_size

        # diffuse part
        rendered["diffuse"] = helper(self.transform_mat_file_diff, transform_mat_diff)

        # specular part
        rendered["specular"] = helper(self.transform_mat_file_spec, transform_mat_spec)

        return rendered


if __name__ == "__main__":
    IBRRender = IBRenderer(ridx=4)
    ibr_diff = IBRRender.transform_mat_diff
    ibr_spec = IBRRender.transform_mat_spec
