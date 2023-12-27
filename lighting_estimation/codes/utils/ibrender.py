"""
This code implements image-based rendering, which is differentiable.
"""
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF

import config as config
from utils.general import read_image, tensor2ndarray, write_image

# self.bases_folder = os.path.join(config.dir_data_files, "")
# self.envmap_size = config.sphere_scale
from pdb import set_trace as st

class IBrender:
    def __init__(self, ridx: int = None, envmap_size: int = None, device: str = "cpu", ibr_setting: str = 'ours'):
    
        self.bases_folder = os.path.join(config.brdf_dir, "")
        self.envmap_size = envmap_size

        self.device = device
        self.ibr_setting = ibr_setting
        self.load_bases(ridx, ibr_setting=self.ibr_setting)

    def load_bases(self, ridx: int = None, ibr_setting: str = 'ours'):
        """Loads bases from .npy files.

        Args:
            envmap_size: size of environment map.
            ridx: index of specular BRDF.
        """
        self.transform_mat_file_diff = os.path.join(self.bases_folder, self.get_diff_name(self.envmap_size) + ".npy")
        transform_mat_diff = np.load(self.transform_mat_file_diff)
        self.transform_mat_diff = TF.to_tensor(transform_mat_diff)[0]

        if ibr_setting == 'ours':
            self.transform_mat_file_spec = os.path.join(self.bases_folder, self.get_spec_name(self.envmap_size, ridx) + ".npy")
        else:
            print("Using Google bases.")
            self.transform_mat_file_spec = os.path.join(self.bases_folder, "google_ms_s64.npy")

        transform_mat_spec = np.load(self.transform_mat_file_spec)
        self.transform_mat_spec = TF.to_tensor(transform_mat_spec)[0]


        if self.device != "cpu":
            self.transform_mat_diff = self.transform_mat_diff.to(self.device)
            self.transform_mat_spec = self.transform_mat_spec.to(self.device)

    def to_cpu(self):
        self.transform_mat_diff = self.transform_mat_diff.to('cpu')
        self.transform_mat_spec = self.transform_mat_spec.to('cpu')

    def to_gpu(self):
        self.transform_mat_diff = self.transform_mat_diff.to(self.device)
        self.transform_mat_spec = self.transform_mat_spec.to(self.device)


    def get_diff_name(self, envmap_size: int):
        """A rememberer of directory name for diffuse BRDF."""
        return f"diff_s{envmap_size}"


    def get_spec_name(self, envmap_size: int,
                    roughness_idx: int):
        """A rememberer of directory name for glossy BRDF."""
        return f"spec_r{roughness_idx:02d}_s{envmap_size}"

    # def get_spec_name(self, envmap_size: int,
    #                 roughness_idx: int):
    #     """A rememberer of directory name for glossy BRDF."""
    #     return f"spec_r{roughness_idx:02d}_s{envmap_size}"


    def grid_visualize(self, sample_interval: int = 4):
        """Visualizes reflectance fields as grids, and writes them to files.
        
        Args:
            sample_interval: sample interval when collecting bases in a reflectance field.
        """
        def helper(folder):
            out_hw = self.envmap_size // sample_interval * self.envmap_size
            grid = np.zeros(shape=(out_hw, out_hw, 3))
            for idx_h in range(sample_interval // 2, self.envmap_size, sample_interval):
                for idx_w in range(sample_interval // 2, self.envmap_size, sample_interval):
                    start_h = idx_h // sample_interval * self.envmap_size
                    start_w = idx_w // sample_interval * self.envmap_size
                    grid[start_h: start_h + self.envmap_size, start_w: start_w + self.envmap_size, :] = \
                        read_image(os.path.join(folder, f"{idx_h:03d}_{idx_w:03d}.hdr"))
            write_image(os.path.join(folder, "grid.hdr"), grid)
        # diffuse
        folder = os.path.join(self.bases_folder, self.get_diff_name(self.envmap_size))
        helper(folder)
        # specular BRDF
        for ridx, _ in enumerate(config.spec_rfns_preset):
            folder = os.path.join(self.bases_folder, self.get_spec_name(self.envmap_size, ridx))
            helper(folder)
        

    def bind_rendered_bases(self):
        """Binds rendered reflectance field bases (reflectance under directional light)
        into transform matrices. Generates 1 matrix for each BRDF (diffuse + 10 specular BRDF)
        and saves it to .npy file.
        """
        def helper(folder):
            transform_mat = np.zeros(shape=(self.envmap_size * self.envmap_size, self.envmap_size * self.envmap_size), dtype=np.float32)
            for idx_h in range(self.envmap_size):
                for idx_w in range(self.envmap_size):
                    reflectance = read_image(os.path.join(folder, f"{idx_h:03d}_{idx_w:03d}.hdr"))
                    reflectance = reflectance[:, :, 0].reshape(-1)
                    transform_mat[idx_h * self.envmap_size + idx_w, :] = reflectance
            np.save(os.path.join(self.bases_folder, os.path.basename(folder) + ".npy"), transform_mat)
        # diffuse BRDF
        folder = os.path.join(self.bases_folder, self.get_diff_name(self.envmap_size))
        helper(folder)
        # specular BRDF
        for ridx, _ in enumerate(config.spec_rfns_preset):
            folder = os.path.join(self.bases_folder, self.get_spec_name(self.envmap_size, ridx))
            helper(folder)
        

    def render_using_mirror_sphere(self, sphere_mirror: torch.Tensor):
        """Renders diffuse and specular ball under the input mirror ball as environment map.
        
        Args:
            sphere_mirror: the input mirror ball under which other two ball are rendered.
            ridx: Roughness level, a integer in [0, len(config.roughness_settings) ).
            transform_mat_diff: pre-loaded transform matrix for diffuse ball. If `None`, reload it.
            transform_mat_spec: pre-loaded transform matrix for glossy ball. If `None`, reload it.
        
        Returns:
            Dict[str, torch.Tensor]: a dict, in which "diffuse" contains rendered diffuse ball,
                and "specular" contains glossy ball.
        """
        
        
        def helper(transform_mat_file, transform_mat=None):
            if transform_mat is None:
                transform_mat = np.load(transform_mat_file)
                transform_mat = TF.to_tensor(transform_mat)[0]
            sphere_ibr = (sphere_mirror.reshape(in_b, in_c, -1) @ transform_mat).reshape(in_b, in_c, self.envmap_size, self.envmap_size)
            sphere_ibr /= 256  # calibrated intensity correction coefficient
            return sphere_ibr
    
        # def helper(transform_mat_file, transform_mat=None):
        #     if transform_mat is None:
        #         transform_mat = np.load(transform_mat_file)
        #         transform_mat = TF.to_tensor(transform_mat)[0]
        #     sphere_ibr = (sphere_mirror.reshape(in_c, -1) @ transform_mat).reshape(in_c, self.envmap_size, self.envmap_size)
        #     sphere_ibr /= 256  # calibrated intensity correction coefficient
        #     return sphere_ibr

        rendered = {}
        in_b, in_c, in_h, in_w = sphere_mirror.shape
        assert in_h == self.envmap_size and in_w == self.envmap_size
        
        # diffuse part
        # transform_mat_file = os.path.join(self.bases_folder, self.get_diff_name(self.envmap_size) + ".npy")
        # rendered["diffuse"] = helper(transform_mat_file, transform_mat_diff)
        
        # # specular part
        # transform_mat_file = os.path.join(self.bases_folder, self.get_spec_name(self.envmap_size, ridx) + ".npy")
        # rendered["specular"] = helper(transform_mat_file, transform_mat_spec)
        
        # diffuse part
        rendered["diffuse"] = helper(self.transform_mat_file_diff, self.transform_mat_diff)
        
        # specular part
        rendered["specular"] = helper(self.transform_mat_file_spec, self.transform_mat_spec)
        
        return rendered

    
    
if __name__ == '__main__':
    # mask = read_image(os.path.join(self.bases_folder, "mask.png"))[:, :, None]

    # grid_visualize()
    # bind_rendered_bases()

    # envmap_folder = os.path.join(config.dir_examples, "envmap", "4")

    # envmap_rect = read_image(r"D:\AlbedoRecon\example\envmap\20130624-161133-P.hdr")
    # envmap = read_image(os.path.join(envmap_folder, "mirror.hdr"))
    # # envmap512 = read_image(os.path.join(envmap_folder, "mirror512.hdr"))
    # envmap_tensor = TF.to_tensor(envmap)
    # rendered = render_using_mirror_sphere(envmap_tensor, ridx=0)
    # write_image(os.path.join(envmap_folder, "diff_ibr.hdr"), tensor2ndarray(rendered["diffuse"]))
    # write_image(os.path.join(envmap_folder, "spec_r0_ibr.hdr"), tensor2ndarray(rendered["specular"]))
    pass