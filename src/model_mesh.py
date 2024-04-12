import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat

from src.utils.train_util import instantiate_from_config


# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        input_size=256,
        render_size=512,
        init_ckpt=None,
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size

        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

        # Load weights from pretrained MVRecon model, and use the mlp 
        # weights to initialize the weights of sdf and rgb mlps.
        if init_ckpt is not None:
            sd = torch.load(init_ckpt, map_location='cpu')['state_dict']
            sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
            sd_fc = {}
            for k, v in sd.items():
                if k.startswith('lrm_generator.synthesizer.decoder.net.'):
                    if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):    # last layer
                        # Here we assume the density filed's isosurface threshold is t, 
                        # we reverse the sign of density filed to initialize SDF field.  
                        # -(w*x + b - t) = (-w)*x + (t - b)
                        if 'weight' in k:
                            sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1]
                        else:
                            sd_fc[k.replace('net.', 'net_sdf.')] = 10.0 - v[0:1]
                        sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                    else:
                        sd_fc[k.replace('net.', 'net_sdf.')] = v
                        sd_fc[k.replace('net.', 'net_rgb.')] = v
                else:
                    sd_fc[k] = v
            sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
            # missing `net_deformation` and `net_weight` parameters
            self.lrm_generator.load_state_dict(sd_fc, strict=False)
            print(f'Loaded weights from {init_ckpt}')
        
        self.validation_step_outputs = []
    
    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.lrm_generator.init_flexicubes_geometry(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        lrm_generator_input = {}
        render_gt = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        # input cameras and render cameras
        input_c2ws = batch['input_c2ws']
        input_Ks = batch['input_Ks']
        target_c2ws = batch['target_c2ws']

        render_c2ws = torch.cat([input_c2ws, target_c2ws], dim=1)
        render_w2cs = torch.linalg.inv(render_c2ws)

        input_extrinsics = input_c2ws.flatten(-2)
        input_extrinsics = input_extrinsics[:, :, :12]
        input_intrinsics = input_Ks.flatten(-2)
        input_intrinsics = torch.stack([
            input_intrinsics[:, :, 0], input_intrinsics[:, :, 4], 
            input_intrinsics[:, :, 2], input_intrinsics[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        # add noise to input_cameras
        cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02

        lrm_generator_input['cameras'] = cameras.to(self.device)
        lrm_generator_input['render_cameras'] = render_w2cs.to(self.device)

        # target images
        target_images = torch.cat([batch['input_images'], batch['target_images']], dim=1)
        target_depths = torch.cat([batch['input_depths'], batch['target_depths']], dim=1)
        target_alphas = torch.cat([batch['input_alphas'], batch['target_alphas']], dim=1)
        target_normals = torch.cat([batch['input_normals'], batch['target_normals']], dim=1)

        render_size = self.render_size
        target_images = v2.functional.resize(
            target_images, render_size, interpolation=3, antialias=True).clamp(0, 1)
        target_depths = v2.functional.resize(
            target_depths, render_size, interpolation=0, antialias=True)
        target_alphas = v2.functional.resize(
            target_alphas, render_size, interpolation=0, antialias=True)
        target_normals = v2.functional.resize(
            target_normals, render_size, interpolation=3, antialias=True)

        lrm_generator_input['render_size'] = render_size

        render_gt['target_images'] = target_images.to(self.device)
        render_gt['target_depths'] = target_depths.to(self.device)
        render_gt['target_alphas'] = target_alphas.to(self.device)
        render_gt['target_normals'] = target_normals.to(self.device)

        return lrm_generator_input, render_gt
    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        # input cameras
        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)

        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input['cameras'] = cameras.to(self.device)

        # render cameras
        render_c2ws = batch['render_c2ws']
        render_w2cs = torch.linalg.inv(render_c2ws)

        lrm_generator_input['render_cameras'] = render_w2cs.to(self.device)
        lrm_generator_input['render_size'] = 384

        return lrm_generator_input
    
    def forward_lrm_generator(self, images, cameras, render_cameras, render_size=512):
        planes = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        out = self.lrm_generator.forward_geometry(
            planes, 
            render_cameras, 
            render_size,
        )
        return out
    
    def forward(self, lrm_generator_input):
        images = lrm_generator_input['images']
        cameras = lrm_generator_input['cameras']
        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']

        out = self.forward_lrm_generator(
            images, cameras, render_cameras, render_size=render_size)

        return out

    def training_step(self, batch, batch_idx):
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)

        render_out = self.forward(lrm_generator_input)

        loss, loss_dict = self.compute_loss(render_out, render_gt)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            B, N, C, H, W = render_gt['target_images'].shape
            N_in = lrm_generator_input['images'].shape[1]

            target_images = rearrange(
                render_gt['target_images'], 'b n c h w -> b c h (n w)')
            render_images = rearrange(
                render_out['img'], 'b n c h w -> b c h (n w)')
            target_alphas = rearrange(
                repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_alphas = rearrange(
                repeat(render_out['mask'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            target_depths = rearrange(
                repeat(render_gt['target_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_depths = rearrange(
                repeat(render_out['depth'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            target_normals = rearrange(
                render_gt['target_normals'], 'b n c h w -> b c h (n w)')
            render_normals = rearrange(
                render_out['normal'], 'b n c h w -> b c h (n w)')
            MAX_DEPTH = torch.max(target_depths)
            target_depths = target_depths / MAX_DEPTH * target_alphas
            render_depths = render_depths / MAX_DEPTH

            grid = torch.cat([
                target_images, render_images, 
                target_alphas, render_alphas, 
                target_depths, render_depths, 
                target_normals, render_normals,
            ], dim=-2)
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))

            image_path = os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png')
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        return loss
    
    def compute_loss(self, render_out, render_gt):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        render_images = render_out['img']
        target_images = render_gt['target_images'].to(render_images)
        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)

        render_alphas = render_out['mask']
        target_alphas = render_gt['target_alphas']
        loss_mask = F.mse_loss(render_alphas, target_alphas)

        render_depths = render_out['depth']
        target_depths = render_gt['target_depths']
        loss_depth = 0.5 * F.l1_loss(render_depths[target_alphas>0], target_depths[target_alphas>0])

        render_normals = render_out['normal'] * 2.0 - 1.0
        target_normals = render_gt['target_normals'] * 2.0 - 1.0
        similarity = (render_normals * target_normals).sum(dim=-3).abs()
        normal_mask = target_alphas.squeeze(-3)
        loss_normal = 1 - similarity[normal_mask>0].mean()
        loss_normal = 0.2 * loss_normal

        # flexicubes regularization loss
        sdf = render_out['sdf']
        sdf_reg_loss = render_out['sdf_reg_loss']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.lrm_generator.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        loss = loss_mse + loss_lpips + loss_mask + loss_normal + loss_reg

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss_mask': loss_mask})
        loss_dict.update({f'{prefix}/loss_normal': loss_normal})
        loss_dict.update({f'{prefix}/loss_depth': loss_depth})
        loss_dict.update({f'{prefix}/loss_reg_sdf': sdf_reg_loss_entropy})
        loss_dict.update({f'{prefix}/loss_reg_surface': flexicubes_surface_reg})
        loss_dict.update({f'{prefix}/loss_reg_weights': flexicubes_weights_reg})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lrm_generator_input = self.prepare_validation_batch_data(batch)

        render_out = self.forward(lrm_generator_input)
        render_images = render_out['img']
        render_images = rearrange(render_images, 'b n c h w -> b c h (n w)')

        self.validation_step_outputs.append(render_images)
    
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=-1)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(
            self.lrm_generator.parameters(), lr=lr, betas=(0.90, 0.95), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100000, eta_min=0)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}