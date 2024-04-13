import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download


def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model")
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def generate_mvs(input_image, sample_steps, sample_seed):

    seed_everything(sample_seed)
    
    # sampling
    generator = torch.Generator(device=device)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image

def make_mesh(mesh_fpath, planes):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_vis_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
        
    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        vertices[:, -1] *= -1
        faces = faces[:, [2, 1, 0]]

        save_obj(vertices, faces, vertex_colors, mesh_fpath)
        
        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath

def make3d(images):

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)

    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384
        
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath



import gradio as gr

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>
'''

_LINKS_ = '''
<h3>Code is available at <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a></h3>
<h3>Report is available at <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a></h3>
'''

_CITE_ = r"""
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        width=379,
                        interactive=False
                    )

                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    width=768,
                    interactive=False,
                )
            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_LINKS_)
    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj]
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839)
