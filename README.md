<div align="center">
  
# InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

<a href="https://huggingface.co/papers/xxx"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-Huggingface-orange"></a> <a href="https://huggingface.co/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/TencentARC/InstantMesh"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

</div>

---

This repo is the official implementation of InstantMesh, a feed-forward framework for efficient 3D mesh generation from a single image. We will release all the code, weights, and demo here.


https://github.com/TencentARC/InstantMesh/assets/20635237/737bba2d-df45-4707-8557-1dd84f248764

# 🚩 Todo List

- [x] Release inference and training code.
- [x] Release model weights.
- [ ] Release hugging face gradio demo (we are waiting for the GPU grant and will make it available as soon as possible).
- [ ] Add support to more multi-view diffusion models.

# ⚙️ Dependencies and Installation

We recommand using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA=12.1`.
```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Install other requirements
pip install -r requirements.txt
```

# 💫 How to Use

## Download the models

We provide 4 sparse-view reconstruction model variants and a customized Zero123++ UNet for white-background image generation in the [model card](https://huggingface.co/TencentARC/InstantMesh).

Please download the models and put them under the `ckpts/` directory.

By default, we use the `instant-mesh-large` reconstruction model variant.

## Start a local gradio demo

To start a gradio demo in your local machine, simply running:
```bash
python app.py
```

## Running with command line

To generate 3D meshes from images via command line, simply running:
```bash
python run.py configs/instant-mesh-large.yaml examples/ --save_video
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object. If the input image already has an alpha mask, please specify the `no_rembg` flag:
```bash
python run.py configs/instant-mesh-large.yaml examples/ --save_video --no_rembg
```

By default, our script exports a `.obj` mesh with vertex colors, please specify the `--export_texmap` flag if you hope to export a mesh with a texture map instead (this will cost longer time):
```bash
python run.py configs/instant-mesh-large.yaml examples/ --save_video --export_texmap
```

Please use a different `.yaml` config file in the [configs](./configs) directory if you hope to use other reconstruction model variants. For example, using the `instant-nerf-large` model for generation:
```bash
python run.py configs/instant-nerf-large.yaml examples/ --save_video
```
**Note:** When using the `NeRF` model variants for image-to-3D generation, exporting a mesh with texture map by specifying `--export_texmap` may cost long time in the UV unwarping step since the default iso-surface extraction resolution is `256`. You can set a lower iso-surface extraction resolution in the config file.

# 💻 Training

We provide our training code to facilatate future research. But we cannot provide the training dataset due to its size. Please refer to our [dataloader](src/data/objaverse.py) for more details.

To train the sparse-view reconstruction models, please run:
```bash
# Training on NeRF representation
python train.py --base configs/instant-nerf-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1

# Training on Mesh representation
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Jiale Xu and Weihao Cheng and Yiming Gao and Xintao Wang and Shenghua Gao and Ying Shan},
  journal={arXiv preprint},
  year={2024}
}
```

# 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [Instant3D](https://instant-3d.github.io/)
