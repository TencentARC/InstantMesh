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


class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        lrm_path=None,
        input_size=256,
        render_size=192,
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size

        # init modules
        self.lrm_generator = instantiate_from_config(lrm_generator_config)
        if lrm_path is not None:
            lrm_ckpt = torch.load(lrm_path)
            self.lrm_generator.load_state_dict(lrm_ckpt['weights'], strict=False)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        
        self.validation_step_outputs = []
    
    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        lrm_generator_input = {}
        render_gt = {}   # for supervision

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        # input cameras and render cameras
        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)
        target_c2ws = batch['target_c2ws'].flatten(-2)
        target_Ks = batch['target_Ks'].flatten(-2)
        render_cameras_input = torch.cat([input_c2ws, input_Ks], dim=-1)
        render_cameras_target = torch.cat([target_c2ws, target_Ks], dim=-1)
        render_cameras = torch.cat([render_cameras_input, render_cameras_target], dim=1)

        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        # add noise to input cameras
        cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02

        lrm_generator_input['cameras'] = cameras.to(self.device)
        lrm_generator_input['render_cameras'] = render_cameras.to(self.device)

        # target images
        target_images = torch.cat([batch['input_images'], batch['target_images']], dim=1)
        target_depths = torch.cat([batch['input_depths'], batch['target_depths']], dim=1)
        target_alphas = torch.cat([batch['input_alphas'], batch['target_alphas']], dim=1)

        # random crop
        render_size = np.random.randint(self.render_size, 513)
        target_images = v2.functional.resize(
            target_images, render_size, interpolation=3, antialias=True).clamp(0, 1)
        target_depths = v2.functional.resize(
            target_depths, render_size, interpolation=0, antialias=True)
        target_alphas = v2.functional.resize(
            target_alphas, render_size, interpolation=0, antialias=True)

        crop_params = v2.RandomCrop.get_params(
            target_images, output_size=(self.render_size, self.render_size))
        target_images = v2.functional.crop(target_images, *crop_params)
        target_depths = v2.functional.crop(target_depths, *crop_params)[:, :, 0:1]
        target_alphas = v2.functional.crop(target_alphas, *crop_params)[:, :, 0:1]

        lrm_generator_input['render_size'] = render_size
        lrm_generator_input['crop_params'] = crop_params

        render_gt['target_images'] = target_images.to(self.device)
        render_gt['target_depths'] = target_depths.to(self.device)
        render_gt['target_alphas'] = target_alphas.to(self.device)

        return lrm_generator_input, render_gt
    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}

        # input images
        images = batch['input_images']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)

        lrm_generator_input['images'] = images.to(self.device)

        input_c2ws = batch['input_c2ws'].flatten(-2)
        input_Ks = batch['input_Ks'].flatten(-2)

        input_extrinsics = input_c2ws[:, :, :12]
        input_intrinsics = torch.stack([
            input_Ks[:, :, 0], input_Ks[:, :, 4], 
            input_Ks[:, :, 2], input_Ks[:, :, 5],
        ], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input['cameras'] = cameras.to(self.device)

        render_c2ws = batch['render_c2ws'].flatten(-2)
        render_Ks = batch['render_Ks'].flatten(-2)
        render_cameras = torch.cat([render_c2ws, render_Ks], dim=-1)

        lrm_generator_input['render_cameras'] = render_cameras.to(self.device)
        lrm_generator_input['render_size'] = 384
        lrm_generator_input['crop_params'] = None

        return lrm_generator_input
    
    def forward_lrm_generator(
        self, 
        images, 
        cameras, 
        render_cameras, 
        render_size=192, 
        crop_params=None, 
        chunk_size=1,
    ):
        planes = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        frames = []
        for i in range(0, render_cameras.shape[1], chunk_size):
            frames.append(
                torch.utils.checkpoint.checkpoint(
                    self.lrm_generator.synthesizer,
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size, 
                    crop_params=crop_params,
                    use_reentrant=False
                )
            )
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        return frames
    
    def forward(self, lrm_generator_input):
        images = lrm_generator_input['images']
        cameras = lrm_generator_input['cameras']
        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']
        crop_params = lrm_generator_input['crop_params']

        out = self.forward_lrm_generator(
            images, 
            cameras, 
            render_cameras, 
            render_size=render_size, 
            crop_params=crop_params, 
            chunk_size=1,
        )
        render_images = torch.clamp(out['images_rgb'], 0.0, 1.0)
        render_depths = out['images_depth']
        render_alphas = torch.clamp(out['images_weight'], 0.0, 1.0)

        out = {
            'render_images': render_images,
            'render_depths': render_depths,
            'render_alphas': render_alphas,
        }
        return out

    def training_step(self, batch, batch_idx):
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)

        render_out = self.forward(lrm_generator_input)

        loss, loss_dict = self.compute_loss(render_out, render_gt)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            B, N, C, H, W = render_gt['target_images'].shape
            N_in = lrm_generator_input['images'].shape[1]

            input_images = v2.functional.resize(
                lrm_generator_input['images'], (H, W), interpolation=3, antialias=True).clamp(0, 1)
            input_images = torch.cat(
                [input_images, torch.ones(B, N-N_in, C, H, W).to(input_images)], dim=1)

            input_images = rearrange(
                input_images, 'b n c h w -> b c h (n w)')
            target_images = rearrange(
                render_gt['target_images'], 'b n c h w -> b c h (n w)')
            render_images = rearrange(
                render_out['render_images'], 'b n c h w -> b c h (n w)')
            target_alphas = rearrange(
                repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_alphas = rearrange(
                repeat(render_out['render_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            target_depths = rearrange(
                repeat(render_gt['target_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_depths = rearrange(
                repeat(render_out['render_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            MAX_DEPTH = torch.max(target_depths)
            target_depths = target_depths / MAX_DEPTH * target_alphas
            render_depths = render_depths / MAX_DEPTH

            grid = torch.cat([
                input_images, 
                target_images, render_images, 
                target_alphas, render_alphas, 
                target_depths, render_depths,
            ], dim=-2)
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))

            save_image(grid, os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png'))

        return loss
    
    def compute_loss(self, render_out, render_gt):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        render_images = render_out['render_images']
        target_images = render_gt['target_images'].to(render_images)
        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0

        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)

        render_alphas = render_out['render_alphas']
        target_alphas = render_gt['target_alphas']
        loss_mask = F.mse_loss(render_alphas, target_alphas)

        loss = loss_mse + loss_lpips + loss_mask

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss_mask': loss_mask})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lrm_generator_input = self.prepare_validation_batch_data(batch)

        render_out = self.forward(lrm_generator_input)
        render_images = render_out['render_images']
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

        params = []

        params.append({"params": self.lrm_generator.parameters(), "lr": lr, "weight_decay": 0.01 })

        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.90, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/10)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
