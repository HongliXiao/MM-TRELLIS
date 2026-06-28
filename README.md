<h1 align="center">[ICRA 2026] MM-TRELLIS: Point-Cloud Guided Multi-Modal 3D Vehicle Generation in Autonomous Driving</h1>

<p align="center">
<a href="https://arxiv.org/abs/2606.24301v1"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://honglixiao.github.io/mm-trellis.github.io/"><img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page"></a>
<a href="https://github.com/microsoft/TRELLIS"><img src="https://img.shields.io/badge/Base_Model-TRELLIS-blue" alt="Base Model"></a>
</p>

<p align="center">
  <img src="assets/comparison_pipeline.png" width="55%">
</p>

<span style="font-size: 16px; font-weight: 600;">MM-TRELLIS</span> is a point-cloud guided multi-modal 3D vehicle generation framework for autonomous driving. Built upon [TRELLIS](https://github.com/microsoft/TRELLIS), MM-TRELLIS extends the original image-to-3D generation pipeline to in-the-wild driving scenarios by incorporating **multi-view image conditioning**, **LiDAR / point-cloud guided sparse-structure generation**, and **opacity-based voxel filtering**. It enables zero-shot vehicle reconstruction from multi-view camera observations and instance-level LiDAR point clouds without retraining the original TRELLIS model.

***Check out our [Project Page](https://honglixiao.github.io/mm-trellis.github.io/) for more visual results and method details!***

<!-- Features -->
## 🌟 Features

- **Multi-View Conditioning**: Uses multiple vehicle views during denoising to aggregate complementary visual information.
- **Point-Cloud Guided Generation**: Incorporates LiDAR point clouds as test-time geometric guidance for sparse-structure generation.
- **Geometry-Aware Vehicle Reconstruction**: Improves shape completeness and geometric consistency in autonomous-driving scenes.
- **Opacity-Based Voxel Filtering**: Uses 3D Gaussian opacity to remove floaters and refine the SLAT features before mesh decoding.
- **Zero-Shot Extension of TRELLIS**: Works on top of the pretrained TRELLIS image model without retraining.

<!-- Overview -->
## 🔍 Method Overview

<p align="center">
  <img src="assets/method_overview.png" width="100%">
</p>

In stage I, we generate voxels with multi-view cycle conditioning and point cloud guidance. The LiDAR point cloud is first preprocessed and rotated by a learnable parameter $\hat{R}$ to a aligned orientation. Then the voxel guidance is applied to optimize the sampled latent during the denoising process.
Stage II perform a 3DGS generation with multi-view conditioning, ensuring the texture fidelity of the generation.
Finally, opacity-based mesh refinement is performed in Stage III: a voxel mask is obtained by thresholding Gaussian opacity and filtering SLAT features, and decoding the filtered features produces the final clean mesh.
   

<!-- The comparison below illustrates the difference between the original TRELLIS pipeline and MM-TRELLIS. The added multi-view conditioning and point-cloud guidance help recover more complete and geometry-consistent vehicles.

<p align="center">
  <img src="assets/comparison_pipeline.png" width="85%">
</p> -->

<!-- Updates -->
## ⏩ Updates

**06/28/2026**
- Release the MM-TRELLIS inference code.

<!-- Installation -->
## 📦 Installation

### Verified Environment

MM-TRELLIS has been tested with the following H20 + CUDA 12.2 environment:

| Component | Version |
| --- | --- |
| GPU | NVIDIA H20 |
| Driver | 535.161.07 |
| System CUDA | 12.2 |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu126 |
| TorchVision | 0.22.1+cu126 |
| xFormers | 0.0.31.post1 |
| spconv | spconv-cu126 2.3.8 |
| cumm | cumm-cu126 0.7.11 |
| PyTorch3D | 0.7.9 |
| Open3D | 0.19.0 |
| Kaolin | 0.18.0 |
| nvdiffrast | 0.3.3 |
| utils3d | 0.0.2 |

The current release is mainly verified on this environment. Other CUDA / PyTorch combinations may also work, but CUDA extensions such as `spconv`, `kaolin`, `nvdiffrast`, `pytorch3d`, and mip-splatting can be sensitive to version changes.

### Prerequisites

- **System**: Linux is recommended.
- **Hardware**: An NVIDIA GPU is required. The released example has been verified on NVIDIA H20.
- **Software**:
  - Python 3.12.3.
  - CUDA 12.2 driver environment.
  - PyTorch 2.7.1 with CUDA 12.x wheels.
  - Conda or Miniconda is recommended for environment management.
  - We recommend using `xformers` as the attention backend.

### Installation Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/HongliXiao/MM-TRELLIS.git
    cd MM-TRELLIS
    ```

2. Create a Python 3.12 environment:

    ```sh
    conda create -n mm-trellis python=3.12 -y
    conda activate mm-trellis
    ```

3. Install PyTorch and xFormers. The verified environment uses PyTorch `2.7.1+cu126`:

    ```sh
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu126

    pip install xformers==0.0.31.post1
    ```

4. Install core Python dependencies:

    ```sh
    pip install \
      accelerate==1.12.0 \
      diffusers==0.36.0 \
      easydict==1.13 \
      huggingface-hub==0.36.0 \
      imageio==2.37.3 \
      imageio-ffmpeg==0.6.0 \
      matplotlib==3.10.8 \
      numpy==2.2.6 \
      open3d==0.19.0 \
      opencv-python-headless==4.12.0.88 \
      pandas==2.3.3 \
      pillow==12.0.0 \
      rembg==2.0.69 \
      safetensors==0.7.0 \
      scikit-image==0.26.0 \
      scikit-learn==1.8.0 \
      scipy==1.16.3 \
      tqdm==4.67.1 \
      transformers==4.57.3 \
      trimesh==4.10.1 \
      utils3d==0.0.2
    ```

5. Install CUDA-related 3D dependencies. The following versions are used in the verified H20 environment:

    ```sh
    pip install spconv-cu126==2.3.8 cumm-cu126==0.7.11
    pip install nvdiffrast==0.3.3
    pip install pytorch3d==0.7.9
    ```

    `kaolin` and mip-splatting / Gaussian rasterization components may need to be installed from source or through the provided TRELLIS setup utilities, depending on your base image. In our verified environment, the following packages are available:

    ```text
    kaolin==0.18.0
    diff_gaussian_rasterization==0.0.0
    ```

6. Set the attention backend:

    ```sh
    export ATTN_BACKEND=xformers
    ```

7. Verify the environment:

    ```sh
    python - <<'PY'
    import torch
    import torchvision
    import xformers
    import open3d
    import pytorch3d
    import utils3d

    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("xformers:", xformers.__version__)
    print("open3d:", open3d.__version__)
    print("utils3d:", utils3d.__version__ if hasattr(utils3d, "__version__") else "installed")
    PY
    ```


<!-- Pretrained Models -->
## 🤖 Pretrained Model

MM-TRELLIS uses the pretrained TRELLIS image-to-3D model ([Hugging Face](https://huggingface.co/microsoft/TRELLIS-image-large)) as the base generator.

<!-- | Model | Description | Download |
| --- | --- | --- |
| TRELLIS-image-large | Large image-to-3D model used by MM-TRELLIS | [Hugging Face](https://huggingface.co/microsoft/TRELLIS-image-large) | -->

You can directly load the model from Hugging Face:

```python
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```

If you prefer loading the model locally, download the model files and pass the local folder path:

```python
TrellisImageTo3DPipeline.from_pretrained("/path/to/TRELLIS-image-large")
```

<!-- Usage -->
## 💡 Usage

### Quick Start

Run the bundled example:

```sh
bash scripts/run_example.sh
```

Equivalent command:

```sh
python inference.py \
  --input_root examples_input/initial_test_instances \
  --pts_root examples_input/drivestudio_way_processed \
  --instances static_006_002 \
  --split test \
  --pretrained_model microsoft/TRELLIS-image-large \
  --output_dir outputs/mm_trellis_example
```

After running inference, the outputs will be saved to:

```text
outputs/mm_trellis_example/<instance>/
├── output.mp4
├── output.glb
├── output.ply
└── output-voxels_out-*.ply
```

The generated files include:
- `output.mp4`: rendered visualization of the generated 3D asset.
- `output.glb`: textured mesh exported as a GLB file.
- `output.ply`: 3D Gaussian representation.
- `output-voxels_out-*.ply`: sparse voxel point cloud produced by the sparse-structure stage.

### Input Data Format

The default resolver expects the following input structure:

```text
examples_input/
├── initial_test_instances/static_006_002/test/*.png
└── drivestudio_way_processed/006/static_006_002/aggregated_instance_lidar_pts/2.symmetric-DBSCAN.ply
```

For an instance named `static_006_002`, the script infers the scene id `006` and loads the point cloud from:

```text
--pts_root/006/static_006_002/aggregated_instance_lidar_pts/2.symmetric-DBSCAN.ply
```

See [`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md) for more details.

### Common Options

Run:

```sh
python inference.py --help
```

Useful arguments:

| Argument | Description |
| --- | --- |
| `--pretrained_model` | Hugging Face repository id or local checkpoint path. |
| `--input_root` | Root directory containing instance folders and image splits. |
| `--pts_root` | Root directory containing preprocessed LiDAR point clouds. |
| `--instances` | One or more instance ids. If omitted, all instances under `--input_root` are processed. |
| `--point_normalization` | Point-cloud normalization mode: `scale-only`, `unit-box`, or `none`. |
| `--cfg` | Classifier-free guidance strength for sparse-structure sampling. |
| `--weight` | Weight for point-cloud guidance loss. |
| `--lr_sample` | Learning rate for latent sample optimization. |
| `--lr_axis_angle` | Rotation learning-rate triple, e.g. `0.001 0.001 0.5`. |
| `--step` | Number of sparse-structure denoising steps. |
| `--min_views`, `--max_views` | Control the number of input views used for multi-view inference. |
| `--image_mode` | Image loading mode: `unchanged`, `RGB`, or `RGBA`. |

<!-- Code Structure -->
## 🧩 Code Structure

```text
MM-TRELLIS/
├── inference.py                   # Main MM-TRELLIS inference script
├── trellis/                       # TRELLIS modules with MM-TRELLIS inference changes
├── examples_input/                # Example multi-view images and point-cloud structure
├── assets/                        # Figures used in this README
├── scripts/run_example.sh         # One-command example
├── docs/                          # Input format, code structure, and license notes
├── setup.sh                       # Dependency installation script
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

Key modified components:

- `inference.py`: Loads multi-view images and LiDAR point clouds, runs MM-TRELLIS inference, and exports results.
- `trellis/pipelines/trellis_image_to_3d.py`: Adds the multi-view point-cloud guided inference pipeline.
- `trellis/pipelines/samplers/flow_euler.py`: Adds point-cloud guided sparse-structure sampling.
- `docs/INPUT_FORMAT.md`: Describes the expected input data structure.

<!-- Notes -->
## 📝 Notes

- MM-TRELLIS is designed as a zero-shot inference extension of TRELLIS.
- This repository focuses on autonomous-driving vehicle generation with multi-view images and LiDAR point clouds.
- The default checkpoint is `microsoft/TRELLIS-image-large`.
- If you use this code, please also cite the original TRELLIS paper and repository.

<!-- License -->
## ⚖️ License and Data Notice

This repository builds on TRELLIS and includes third-party components with their own license headers. Before public release, review [`docs/LICENSE_NOTICE.md`](docs/LICENSE_NOTICE.md) and choose a top-level license that is compatible with all included code and data.

<!-- The bundled example data follows a Waymo / DriveStudio-style format. If it is derived from the Waymo Open Dataset, redistribution and downstream use must comply with the Waymo Open Dataset terms. See [`examples_input/README.md`](examples_input/README.md). -->

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing MM-TRELLIS:

```bibtex
@misc{xiao2026mmtrellis,
  title         = {MM-TRELLIS: Point-Cloud Guided Multi-Modal 3D Vehicle Generation in Autonomous Driving},
  author        = {Xiao, Hongli and Zhang, Youjian and Bai, Yucai and Wang, Chaoyue and Jin, Yaohui and Ren, Xiaoguang and Yang, Wenjing and Lan, Long},
  year          = {2026},
  eprint        = {2606.24301},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

Please also cite TRELLIS if you use the base model or code:

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```
