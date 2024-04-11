# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from . import Renderer

_FG_LUT = None


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def compute_vertex_normal(v_pos, t_pos_idx):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)
    assert torch.all(torch.isfinite(v_nrm))

    return v_nrm


class NeuralRender(Renderer):
    def __init__(self, device='cuda', camera_model=None):
        super(NeuralRender, self).__init__()
        self.device = device
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.projection_mtx = None
        self.camera = camera_model

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask
        
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera

        v_nrm = compute_vertex_normal(mesh_v_pos_bxnx3[0], mesh_t_pos_idx_fx3.long())  # vertex normals in world coordinates

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd.repeat(v_pos.shape[0], 1, 1), v_pos], dim=-1)  # Concatenate the pos

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]

        normal, _ = interpolate(v_nrm[None, ...], rast, mesh_t_pos_idx_fx3)
        normal = dr.antialias(normal.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        normal = F.normalize(normal, dim=-1)
        normal = torch.lerp(torch.zeros_like(normal), (normal + 1.0) / 2.0, hard_mask.float())      # black background

        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal
