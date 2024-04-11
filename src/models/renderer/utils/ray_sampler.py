# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified by Jiale Xu
# The modifications are subject to the same license as the original.


"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch

class RaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, intrinsics, render_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        render_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """

        dtype = cam2world_matrix.dtype
        device = cam2world_matrix.device
        N, M = cam2world_matrix.shape[0], render_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(torch.meshgrid(
            torch.arange(render_size, dtype=dtype, device=device),
            torch.arange(render_size, dtype=dtype, device=device),
            indexing='ij',
        ))
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1) * (1./render_size) + (0.5/render_size)
        y_cam = uv[:, :, 1].view(N, -1) * (1./render_size) + (0.5/render_size)
        z_cam = torch.ones((N, M), dtype=dtype, device=device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1).to(dtype)

        _opencv2blender = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)

        cam2world_matrix = torch.bmm(cam2world_matrix, _opencv2blender)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2).to(dtype)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs


class OrthoRaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, ortho_scale, render_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        ortho_scale: float
        render_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 3)
        """

        N, M = cam2world_matrix.shape[0], render_size**2

        uv = torch.stack(torch.meshgrid(
            torch.arange(render_size, dtype=torch.float32, device=cam2world_matrix.device),
            torch.arange(render_size, dtype=torch.float32, device=cam2world_matrix.device),
            indexing='ij',
        ))
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1) * (1./render_size) + (0.5/render_size)
        y_cam = uv[:, :, 1].view(N, -1) * (1./render_size) + (0.5/render_size)
        z_cam = torch.zeros((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - 0.5) * ortho_scale
        y_lift = (y_cam - 0.5) * ortho_scale

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        _opencv2blender = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=cam2world_matrix.device).unsqueeze(0).repeat(N, 1, 1)

        cam2world_matrix = torch.bmm(cam2world_matrix, _opencv2blender)

        ray_origins = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        ray_dirs_cam = torch.stack([
            torch.zeros((N, M), device=cam2world_matrix.device),
            torch.zeros((N, M), device=cam2world_matrix.device),
            torch.ones((N, M), device=cam2world_matrix.device),
        ], dim=-1)
        ray_dirs = torch.bmm(cam2world_matrix[:, :3, :3], ray_dirs_cam.permute(0, 2, 1)).permute(0, 2, 1)

        return ray_origins, ray_dirs
