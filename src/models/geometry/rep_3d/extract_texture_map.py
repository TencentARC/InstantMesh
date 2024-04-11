# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import xatlas
import numpy as np
import nvdiffrast.torch as dr


# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def xatlas_uvmap(ctx, mesh_v, mesh_pos_idx, resolution):
    vmapping, indices, uvs = xatlas.parametrize(mesh_v.detach().cpu().numpy(), mesh_pos_idx.detach().cpu().numpy())

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
    mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
    # mesh_v_tex. ture
    uv_clip = uvs[None, ...] * 2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), (resolution, resolution))

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
    mask = rast[..., 3:4] > 0
    return uvs, mesh_tex_idx, gb_pos, mask
