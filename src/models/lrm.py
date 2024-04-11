# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import mcubes
import nvdiffrast.torch as dr
from einops import rearrange, repeat

from .encoder.dino_wrapper import DinoWrapper
from .decoder.transformer import TriplaneTransformer
from .renderer.synthesizer import TriplaneSynthesizer
from ..utils.mesh_util import xatlas_uvmap


class InstantNeRF(nn.Module):
    """
    Full model of the large reconstruction model.
    """
    def __init__(
        self, 
        encoder_freeze: bool = False, 
        encoder_model_name: str = 'facebook/dino-vitb16', 
        encoder_feat_dim: int = 768,
        transformer_dim: int = 1024, 
        transformer_layers: int = 16, 
        transformer_heads: int = 16,
        triplane_low_res: int = 32, 
        triplane_high_res: int = 64, 
        triplane_dim: int = 80,
        rendering_samples_per_ray: int = 128,
    ):
        super().__init__()
        
        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )

        self.transformer = TriplaneTransformer(
            inner_dim=transformer_dim, 
            num_layers=transformer_layers, 
            num_heads=transformer_heads,
            image_feat_dim=encoder_feat_dim,
            triplane_low_res=triplane_low_res, 
            triplane_high_res=triplane_high_res, 
            triplane_dim=triplane_dim,
        )

        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, 
            samples_per_ray=rendering_samples_per_ray,
        )

    def forward_planes(self, images, cameras):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        B = images.shape[0]

        # encode images
        image_feats = self.encoder(images, cameras)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        
        # transformer generating planes
        planes = self.transformer(image_feats)

        return planes
    
    def forward_synthesizer(self, planes, render_cameras, render_size: int):
        render_results = self.synthesizer(
            planes, 
            render_cameras, 
            render_size,
        )
        return render_results

    def forward(self, images, cameras, render_cameras, render_size: int):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        # render_cameras: [B, M, D_cam_render]
        # render_size: int
        B, M = render_cameras.shape[:2]

        planes = self.forward_planes(images, cameras)

        # render target views
        render_results = self.synthesizer(planes, render_cameras, render_size)

        return {
            'planes': planes,
            **render_results,
        }
    
    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):
        '''
        Predict Texture given triplanes
        :param planes: the triplane feature map
        :param tex_pos: Position we want to query the texture field
        :param hard_mask: 2D silhoueete of the rendered image
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat(
                        [tex_pos_one_shape, torch.zeros(
                            1, max_point - tex_pos_one_shape.shape[1], 3,
                            device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)

        tex_feat = torch.utils.checkpoint.checkpoint(
            self.synthesizer.forward_points, 
            planes, 
            tex_pos,
            use_reentrant=False,
        )['rgb']

        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(planes.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])

    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        mesh_resolution: int = 256, 
        mesh_threshold: int = 10.0, 
        use_texture_map: bool = False, 
        texture_resolution: int = 1024,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from triplane nerf. Only support batch_size 1.
        :param planes: triplane features
        :param mesh_resolution: marching cubes resolution
        :param mesh_threshold: iso-surface threshold
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texture map
        '''
        assert planes.shape[0] == 1
        device = planes.device

        grid_out = self.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_resolution,
        )
        
        vertices, faces = mcubes.marching_cubes(
            grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), 
            mesh_threshold,
        )
        vertices = vertices / (mesh_resolution - 1) * 2 - 1

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
            vertices_colors = self.synthesizer.forward_points(
                planes, vertices_tensor)['rgb'].squeeze(0).cpu().numpy()
            vertices_colors = (vertices_colors * 255).astype(np.uint8)

            return vertices, faces, vertices_colors
        
        # use x-atlas to get uv mapping for the mesh
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(faces.astype(int), dtype=torch.long, device=device)

        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # query the texture field to get the RGB color for texture map
        tex_feat = self.get_texture_prediction(
            planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, texture_map