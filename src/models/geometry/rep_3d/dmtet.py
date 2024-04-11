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
from .dmtet_utils import get_center_boundary_index
import torch.nn.functional as F


###############################################################################
# DMTet utility functions
###############################################################################
def create_mt_variable(device):
    triangle_table = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)

    num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=device)
    base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=device)
    v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=device))
    return triangle_table, num_triangles_table, base_tet_edges, v_id


def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)
        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
    return torch.stack([a, b], -1)


###############################################################################
# marching tetrahedrons (differentiable)
###############################################################################

def marching_tets(pos_nx3, sdf_n, tet_fx4, triangle_table, num_triangles_table, base_tet_edges, v_id):
    with torch.no_grad():
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tet_fx4[valid_tets][:, base_tet_edges].reshape(-1, 2)
        all_edges = sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=sdf_n.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=sdf_n.device)
        idx_map = mapping[idx_map]  # map edges to verts

        interp_v = unique_edges[mask_edges]  # .long()
    edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
    edges_to_interp_sdf[:, -1] *= -1

    denominator = edges_to_interp_sdf.sum(1, keepdim=True)

    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table[tetindex]

    # Generate triangle indices
    faces = torch.cat(
        (
            torch.gather(
                input=idx_map[num_triangles == 1], dim=1,
                index=triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(
                input=idx_map[num_triangles == 2], dim=1,
                index=triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)
    return verts, faces


def create_tetmesh_variables(device='cuda'):
    tet_table = torch.tensor(
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
         [0, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1],
         [1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1],
         [1, 0, 8, 7, 0, 5, 8, 7, 0, 5, 6, 8],
         [2, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1],
         [2, 0, 9, 7, 0, 4, 9, 7, 0, 4, 6, 9],
         [2, 1, 9, 5, 1, 4, 9, 5, 1, 4, 8, 9],
         [6, 0, 1, 2, 6, 1, 2, 8, 6, 8, 2, 9],
         [3, 6, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1],
         [3, 0, 9, 8, 0, 4, 9, 8, 0, 4, 5, 9],
         [3, 1, 9, 6, 1, 4, 9, 6, 1, 4, 7, 9],
         [5, 0, 1, 3, 5, 1, 3, 7, 5, 7, 3, 9],
         [3, 2, 8, 6, 2, 5, 8, 6, 2, 5, 7, 8],
         [4, 0, 2, 3, 4, 2, 3, 7, 4, 7, 3, 8],
         [4, 1, 2, 3, 4, 2, 3, 5, 4, 5, 3, 6],
         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.long, device=device)
    num_tets_table = torch.tensor([0, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 0], dtype=torch.long, device=device)
    return tet_table, num_tets_table


def marching_tets_tetmesh(
        pos_nx3, sdf_n, tet_fx4, triangle_table, num_triangles_table, base_tet_edges, v_id,
        return_tet_mesh=False, ori_v=None, num_tets_table=None, tet_table=None):
    with torch.no_grad():
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tet_fx4[valid_tets][:, base_tet_edges].reshape(-1, 2)
        all_edges = sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=sdf_n.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=sdf_n.device)
        idx_map = mapping[idx_map]  # map edges to verts

        interp_v = unique_edges[mask_edges]  # .long()
    edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
    edges_to_interp_sdf[:, -1] *= -1

    denominator = edges_to_interp_sdf.sum(1, keepdim=True)

    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table[tetindex]

    # Generate triangle indices
    faces = torch.cat(
        (
            torch.gather(
                input=idx_map[num_triangles == 1], dim=1,
                index=triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(
                input=idx_map[num_triangles == 2], dim=1,
                index=triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)
    if not return_tet_mesh:
        return verts, faces
    occupied_verts = ori_v[occ_n]
    mapping = torch.ones((pos_nx3.shape[0]), dtype=torch.long, device="cuda") * -1
    mapping[occ_n] = torch.arange(occupied_verts.shape[0], device="cuda")
    tet_fx4 = mapping[tet_fx4.reshape(-1)].reshape((-1, 4))

    idx_map = torch.cat([tet_fx4[valid_tets] + verts.shape[0], idx_map], -1)  # t x 10
    tet_verts = torch.cat([verts, occupied_verts], 0)
    num_tets = num_tets_table[tetindex]

    tets = torch.cat(
        (
            torch.gather(input=idx_map[num_tets == 1], dim=1, index=tet_table[tetindex[num_tets == 1]][:, :4]).reshape(
                -1,
                4),
            torch.gather(input=idx_map[num_tets == 3], dim=1, index=tet_table[tetindex[num_tets == 3]][:, :12]).reshape(
                -1,
                4),
        ), dim=0)
    # add fully occupied tets
    fully_occupied = occ_fx4.sum(-1) == 4
    tet_fully_occupied = tet_fx4[fully_occupied] + verts.shape[0]
    tets = torch.cat([tets, tet_fully_occupied])

    return verts, faces, tet_verts, tets


###############################################################################
# Compact tet grid
###############################################################################

def compact_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Find surface tets
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)  # one value per tet, these are the surface tets

        valid_vtx = tet_fx4[valid_tets].reshape(-1)
        unique_vtx, idx_map = torch.unique(valid_vtx, dim=0, return_inverse=True)
        new_pos = pos_nx3[unique_vtx]
        new_sdf = sdf_n[unique_vtx]
        new_tets = idx_map.reshape(-1, 4)
        return new_pos, new_sdf, new_tets


###############################################################################
# Subdivide volume
###############################################################################

def batch_subdivide_volume(tet_pos_bxnx3, tet_bxfx4, grid_sdf):
    device = tet_pos_bxnx3.device
    # get new verts
    tet_fx4 = tet_bxfx4[0]
    edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
    all_edges = tet_fx4[:, edges].reshape(-1, 2)
    all_edges = sort_edges(all_edges)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
    idx_map = idx_map + tet_pos_bxnx3.shape[1]
    all_values = torch.cat([tet_pos_bxnx3, grid_sdf], -1)
    mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
        all_values.shape[0], -1, 2,
        all_values.shape[-1]).mean(2)
    new_v = torch.cat([all_values, mid_points_pos], 1)
    new_v, new_sdf = new_v[..., :3], new_v[..., 3]

    # get new tets

    idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
    idx_ab = idx_map[0::6]
    idx_ac = idx_map[1::6]
    idx_ad = idx_map[2::6]
    idx_bc = idx_map[3::6]
    idx_bd = idx_map[4::6]
    idx_cd = idx_map[5::6]

    tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
    tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
    tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
    tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
    tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
    tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
    tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
    tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

    tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
    tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
    tet = tet_np.long().to(device)

    return new_v, tet, new_sdf


###############################################################################
# Adjacency
###############################################################################
def tet_to_tet_adj_sparse(tet_tx4):
    # include self connection!!!!!!!!!!!!!!!!!!!
    with torch.no_grad():
        t = tet_tx4.shape[0]
        device = tet_tx4.device
        idx_array = torch.LongTensor(
            [0, 1, 2,
             1, 0, 3,
             2, 3, 0,
             3, 2, 1]).to(device).reshape(4, 3).unsqueeze(0).expand(t, -1, -1)  # (t, 4, 3)

        # get all faces
        all_faces = torch.gather(input=tet_tx4.unsqueeze(1).expand(-1, 4, -1), index=idx_array, dim=-1).reshape(
            -1,
            3)  # (tx4, 3)
        all_faces_tet_idx = torch.arange(t, device=device).unsqueeze(-1).expand(-1, 4).reshape(-1)
        # sort and group
        all_faces_sorted, _ = torch.sort(all_faces, dim=1)

        all_faces_unique, inverse_indices, counts = torch.unique(
            all_faces_sorted, dim=0, return_counts=True,
            return_inverse=True)
        tet_face_fx3 = all_faces_unique[counts == 2]
        counts = counts[inverse_indices]  # tx4
        valid = (counts == 2)

        group = inverse_indices[valid]
        # print (inverse_indices.shape, group.shape, all_faces_tet_idx.shape)
        _, indices = torch.sort(group)
        all_faces_tet_idx_grouped = all_faces_tet_idx[valid][indices]
        tet_face_tetidx_fx2 = torch.stack([all_faces_tet_idx_grouped[::2], all_faces_tet_idx_grouped[1::2]], dim=-1)

        tet_adj_idx = torch.cat([tet_face_tetidx_fx2, torch.flip(tet_face_tetidx_fx2, [1])])
        adj_self = torch.arange(t, device=tet_tx4.device)
        adj_self = torch.stack([adj_self, adj_self], -1)
        tet_adj_idx = torch.cat([tet_adj_idx, adj_self])

        tet_adj_idx = torch.unique(tet_adj_idx, dim=0)
        values = torch.ones(
            tet_adj_idx.shape[0], device=tet_tx4.device).float()
        adj_sparse = torch.sparse.FloatTensor(
            tet_adj_idx.t(), values, torch.Size([t, t]))

        # normalization
        neighbor_num = 1.0 / torch.sparse.sum(
            adj_sparse, dim=1).to_dense()
        values = torch.index_select(neighbor_num, 0, tet_adj_idx[:, 0])
        adj_sparse = torch.sparse.FloatTensor(
            tet_adj_idx.t(), values, torch.Size([t, t]))
    return adj_sparse


###############################################################################
# Compact grid
###############################################################################

def get_tet_bxfx4x3(bxnxz, bxfx4):
    n_batch, z = bxnxz.shape[0], bxnxz.shape[2]
    gather_input = bxnxz.unsqueeze(2).expand(
        n_batch, bxnxz.shape[1], 4, z)
    gather_index = bxfx4.unsqueeze(-1).expand(
        n_batch, bxfx4.shape[1], 4, z).long()
    tet_bxfx4xz = torch.gather(
        input=gather_input, dim=1, index=gather_index)

    return tet_bxfx4xz


def shrink_grid(tet_pos_bxnx3, tet_bxfx4, grid_sdf):
    with torch.no_grad():
        assert tet_pos_bxnx3.shape[0] == 1

        occ = grid_sdf[0] > 0
        occ_sum = get_tet_bxfx4x3(occ.unsqueeze(0).unsqueeze(-1), tet_bxfx4).reshape(-1, 4).sum(-1)
        mask = (occ_sum > 0) & (occ_sum < 4)

        # build connectivity graph
        adj_matrix = tet_to_tet_adj_sparse(tet_bxfx4[0])
        mask = mask.float().unsqueeze(-1)

        # Include a one ring of neighbors
        for i in range(1):
            mask = torch.sparse.mm(adj_matrix, mask)
        mask = mask.squeeze(-1) > 0

        mapping = torch.zeros((tet_pos_bxnx3.shape[1]), device=tet_pos_bxnx3.device, dtype=torch.long)
        new_tet_bxfx4 = tet_bxfx4[:, mask].long()
        selected_verts_idx = torch.unique(new_tet_bxfx4)
        new_tet_pos_bxnx3 = tet_pos_bxnx3[:, selected_verts_idx]
        mapping[selected_verts_idx] = torch.arange(selected_verts_idx.shape[0], device=tet_pos_bxnx3.device)
        new_tet_bxfx4 = mapping[new_tet_bxfx4.reshape(-1)].reshape(new_tet_bxfx4.shape)
        new_grid_sdf = grid_sdf[:, selected_verts_idx]
        return new_tet_pos_bxnx3, new_tet_bxfx4, new_grid_sdf


###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0],
        (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1],
                   (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


###############################################################################
#  Geometry interface
###############################################################################
class DMTetGeometry(Geometry):
    def __init__(
            self, grid_res=64, scale=2.0, device='cuda', renderer=None,
            render_type='neural_render', args=None):
        super(DMTetGeometry, self).__init__()
        self.grid_res = grid_res
        self.device = device
        self.args = args
        tets = np.load('data/tets/%d_compress.npz' % (grid_res))
        self.verts = torch.from_numpy(tets['vertices']).float().to(self.device)
        # Make sure the tet is zero-centered and length is equal to 1
        length = self.verts.max(dim=0)[0] - self.verts.min(dim=0)[0]
        length = length.max()
        mid = (self.verts.max(dim=0)[0] + self.verts.min(dim=0)[0]) / 2.0
        self.verts = (self.verts - mid.unsqueeze(dim=0)) / length
        if isinstance(scale, list):
            self.verts[:, 0] = self.verts[:, 0] * scale[0]
            self.verts[:, 1] = self.verts[:, 1] * scale[1]
            self.verts[:, 2] = self.verts[:, 2] * scale[1]
        else:
            self.verts = self.verts * scale
        self.indices = torch.from_numpy(tets['tets']).long().to(self.device)
        self.triangle_table, self.num_triangles_table, self.base_tet_edges, self.v_id = create_mt_variable(self.device)
        self.tet_table, self.num_tets_table = create_tetmesh_variables(self.device)
        # Parameters for regularization computation
        edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=self.device)
        all_edges = self.indices[:, edges].reshape(-1, 2)
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)

        # Parameters used for fix boundary sdf
        self.center_indices, self.boundary_indices = get_center_boundary_index(self.verts)
        self.renderer = renderer
        self.render_type = render_type

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def get_mesh(self, v_deformed_nx3, sdf_n, with_uv=False, indices=None):
        if indices is None:
            indices = self.indices
        verts, faces = marching_tets(
            v_deformed_nx3, sdf_n, indices, self.triangle_table,
            self.num_triangles_table, self.base_tet_edges, self.v_id)
        faces = torch.cat(
            [faces[:, 0:1],
             faces[:, 2:3],
             faces[:, 1:2], ], dim=-1)
        return verts, faces

    def get_tet_mesh(self, v_deformed_nx3, sdf_n, with_uv=False, indices=None):
        if indices is None:
            indices = self.indices
        verts, faces, tet_verts, tets = marching_tets_tetmesh(
            v_deformed_nx3, sdf_n, indices, self.triangle_table,
            self.num_triangles_table, self.base_tet_edges, self.v_id, return_tet_mesh=True,
            num_tets_table=self.num_tets_table, tet_table=self.tet_table, ori_v=v_deformed_nx3)
        faces = torch.cat(
            [faces[:, 0:1],
             faces[:, 2:3],
             faces[:, 1:2], ], dim=-1)
        return verts, faces, tet_verts, tets

    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, camera_mv_bx4x4, resolution=256, hierarchical_mask=False):
        return_value = dict()
        if self.render_type == 'neural_render':
            tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth = self.renderer.render_mesh(
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
