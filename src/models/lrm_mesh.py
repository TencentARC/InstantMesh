# Copyright (c) 2023, Tencent Inc
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
import nvdiffrast.torch as dr
from einops import rearrange, repeat

from .encoder.dino_wrapper import DinoWrapper
from .decoder.transformer import TriplaneTransformer
from .renderer.synthesizer_mesh import TriplaneSynthesizer
from .geometry.camera.perspective_camera import PerspectiveCamera
from .geometry.render.neural_render import NeuralRender
from .geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
from ..utils.mesh_util import xatlas_uvmap


class InstantMesh(nn.Module):
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
        grid_res: int = 128, 
        grid_scale: float = 2.0,
    ):
        super().__init__()
        
        # attributes
        self.grid_res = grid_res
        self.grid_scale = grid_scale
        self.deformation_multiplier = 4.0

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

    def init_flexicubes_geometry(self, device, fovy=50.0):
        camera = PerspectiveCamera(fovy=fovy, device=device)
        renderer = NeuralRender(device, camera_model=camera)
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res, 
            scale=self.grid_scale, 
            renderer=renderer, 
            render_type='neural_render',
            device=device,
        )

    def forward_planes(self, images, cameras):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        B = images.shape[0]

        # encode images
        image_feats = self.encoder(images, cameras)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        
        # decode triplanes
        planes = self.transformer(image_feats)

        return planes
    
    def get_sdf_deformation_prediction(self, planes):
        '''
        Predict SDF and deformation for tetrahedron vertices
        :param planes: triplane feature map for the geometry
        '''
        init_position = self.geometry.verts.unsqueeze(0).expand(planes.shape[0], -1, -1)
        
        # Step 1: predict the SDF and deformation
        sdf, deformation, weight = torch.utils.checkpoint.checkpoint(
            self.synthesizer.get_geometry_prediction,
            planes, 
            init_position, 
            self.geometry.indices,
            use_reentrant=False,
        )

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ####
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight
    
    def get_geometry_prediction(self, planes=None):
        '''
        Function to generate mesh with give triplanes
        :param planes: triplane features
        '''
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(planes)
        v_deformed = self.geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        
        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces, flexicubes_surface_reg = self.geometry.get_mesh(
                v_deformed[i_batch], 
                sdf[i_batch].squeeze(dim=-1),
                with_uv=False, 
                indices=tets, 
                weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training,
            )
            flexicubes_surface_reg_list.append(flexicubes_surface_reg)
            v_list.append(verts)
            f_list.append(faces)
        
        flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
        flexicubes_weight_reg = (weight ** 2).mean()
        
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)
    
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
            self.synthesizer.get_texture_prediction,
            planes, 
            tex_pos,
            use_reentrant=False,
        )

        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(planes.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=render_size,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask = torch.cat(return_value['mask'], dim=0)
        hard_mask = torch.cat(return_value['hard_mask'], dim=0)
        tex_pos = return_value['tex_pos']
        depth = torch.cat(return_value['depth'], dim=0)
        normal = torch.cat(return_value['normal'], dim=0)
        return mask, hard_mask, tex_pos, depth, normal
    
    def forward_geometry(self, planes, render_cameras, render_size=256):
        '''
        Main function of our Generator. It first generate 3D mesh, then render it into 2D image
        with given `render_cameras`.
        :param planes: triplane features
        :param render_cameras: cameras to render generated 3D shape
        '''
        B, NV = render_cameras.shape[:2]

        # Generate 3D mesh first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)

        # Render the mesh into 2D image (get 3d position of each image plane)
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(mesh_v, mesh_f, cam_mv, render_size=render_size)

        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(planes.shape[0])], dim=0)

        # Querying the texture field to predict the texture feature for each pixel on the image
        tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        background_feature = torch.ones_like(tex_feat)      # white background

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        # We should split it back to the original image shape
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)

        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        depth = -depth.permute(0, 3, 1, 2).unflatten(0, (B, NV))        # transform negative depth to positive
        normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        out = {
            'img': img,
            'mask': antilias_mask,
            'depth': depth,
            'normal': normal,
            'sdf': sdf,
            'mesh_v': mesh_v,
            'mesh_f': mesh_f,
            'sdf_reg_loss': sdf_reg_loss,
        }
        return out

    def forward(self, images, cameras, render_cameras, render_size: int):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        # render_cameras: [B, M, D_cam_render]
        # render_size: int
        B, M = render_cameras.shape[:2]

        planes = self.forward_planes(images, cameras)
        out = self.forward_geometry(planes, render_cameras, render_size=render_size)

        return {
            'planes': planes,
            **out
        }

    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from FlexiCubes. Only support batch_size 1.
        :param planes: triplane features
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texure map
        '''
        assert planes.shape[0] == 1
        device = planes.device

        # predict geometry first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = vertices.unsqueeze(0)
            vertices_colors = self.synthesizer.get_texture_prediction(
                planes, vertices_tensor).clamp(0, 1).squeeze(0).cpu().numpy()
            vertices_colors = (vertices_colors * 255).astype(np.uint8)

            return vertices.cpu().numpy(), faces.cpu().numpy(), vertices_colors

        # use x-atlas to get uv mapping for the mesh
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # query the texture field to get the RGB color for texture map
        tex_feat = self.get_texture_prediction(
            planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, texture_map