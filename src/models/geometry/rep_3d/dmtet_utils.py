# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


def get_center_boundary_index(verts):
    length_ = torch.sum(verts ** 2, dim=-1)
    center_idx = torch.argmin(length_)
    boundary_neg = verts == verts.max()
    boundary_pos = verts == verts.min()
    boundary = torch.bitwise_or(boundary_pos, boundary_neg)
    boundary = torch.sum(boundary.float(), dim=-1)
    boundary_idx = torch.nonzero(boundary)
    return center_idx, boundary_idx.squeeze(dim=-1)
