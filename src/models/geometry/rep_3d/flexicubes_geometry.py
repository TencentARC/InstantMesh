# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import os
from . import Geometry
from .flexicubes import FlexiCubes # replace later
from .dmtet import sdf_reg_loss_batch
import torch.nn.functional as F

def get_center_boundary_index(grid_res, device):
    v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool, device=device)
    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
    center_indices = torch.nonzero(v.reshape(-1))

    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
    v[:2, ...] = True
    v[-2:, ...] = True
    v[:, :2, ...] = True
    v[:, -2:, ...] = True
    v[:, :, :2] = True
    v[:, :, -2:] = True
    boundary_indices = torch.nonzero(v.reshape(-1))
    return center_indices, boundary_indices

###############################################################################
#  Geometry interface
###############################################################################
class FlexiCubesGeometry(Geometry):
    def __init__(
            self, grid_res=64, scale=2.0, device='cuda', renderer=None,
            render_type='neural_render', args=None):
        super(FlexiCubesGeometry, self).__init__()
        self.grid_res = grid_res
        self.device = device
        self.args = args
        self.fc = FlexiCubes(device, weight_scale=0.5)
        self.verts, self.indices = self.fc.construct_voxel_grid(grid_res)
        if isinstance(scale, list):
            self.verts[:, 0] = self.verts[:, 0] * scale[0]
            self.verts[:, 1] = self.verts[:, 1] * scale[1]
            self.verts[:, 2] = self.verts[:, 2] * scale[1]
        else:
            self.verts = self.verts * scale
            
        all_edges = self.indices[:, self.fc.cube_edges].reshape(-1, 2)
        self.all_edges = torch.unique(all_edges, dim=0)

        # Parameters used for fix boundary sdf
        self.center_indices, self.boundary_indices = get_center_boundary_index(self.grid_res, device)
        self.renderer = renderer
        self.render_type = render_type

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def get_mesh(self, v_deformed_nx3, sdf_n, weight_n=None, with_uv=False, indices=None, is_training=False):
        if indices is None:
            indices = self.indices

        verts, faces, v_reg_loss = self.fc(v_deformed_nx3, sdf_n, indices, self.grid_res,
                                            beta_fx12=weight_n[:, :12], alpha_fx8=weight_n[:, 12:20],
                                            gamma_f=weight_n[:, 20], training=is_training
                                            )
        return verts, faces, v_reg_loss


    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, camera_mv_bx4x4, resolution=256, hierarchical_mask=False):
        return_value = dict()
        if self.render_type == 'neural_render':
            tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal = self.renderer.render_mesh(
                mesh_v_nx3.unsqueeze(dim=0),
                mesh_f_fx3.int(),
                camera_mv_bx4x4,
                mesh_v_nx3.unsqueeze(dim=0),
                resolution=resolution,
                device=self.device,
                hierarchical_mask=hierarchical_mask
            )

            return_value['tex_pos'] = tex_pos
            return_value['mask'] = mask
            return_value['hard_mask'] = hard_mask
            return_value['rast'] = rast
            return_value['v_pos_clip'] = v_pos_clip
            return_value['mask_pyramid'] = mask_pyramid
            return_value['depth'] = depth
            return_value['normal'] = normal
        else:
            raise NotImplementedError

        return return_value

    def render(self, v_deformed_bxnx3=None, sdf_bxn=None, camera_mv_bxnviewx4x4=None, resolution=256):
        # Here I assume a batch of meshes (can be different mesh and geometry), for the other shapes, the batch is 1
        v_list = []
        f_list = []
        n_batch = v_deformed_bxnx3.shape[0]
        all_render_output = []
        for i_batch in range(n_batch):
            verts_nx3, faces_fx3 = self.get_mesh(v_deformed_bxnx3[i_batch], sdf_bxn[i_batch])
            v_list.append(verts_nx3)
            f_list.append(faces_fx3)
            render_output = self.render_mesh(verts_nx3, faces_fx3, camera_mv_bxnviewx4x4[i_batch], resolution)
            all_render_output.append(render_output)

        # Concatenate all render output
        return_keys = all_render_output[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in all_render_output]
            return_value[k] = value
            # We can do concatenation outside of the render
        return return_value
