# Code structure

This repository is organized around the original TRELLIS codebase, with MM-TRELLIS changes concentrated in the image-to-3D pipeline and sparse-structure sampler.

## Main entry point

- `inference.py`: user-facing script for running point-cloud guided vehicle generation.

## Modified / important TRELLIS components

- `trellis/pipelines/trellis_image_to_3d.py`
  - image-conditioned generation pipeline;
  - MM-TRELLIS variants of `run_*_with_grad_*` functions;
  - opacity-based Gaussian/voxel filtering before mesh decoding.

- `trellis/pipelines/samplers/flow_euler.py`
  - rectified-flow Euler sampler;
  - guided sparse-structure denoising;
  - point-cloud / voxel consistency loss;
  - pose/rotation alignment support.

- `trellis/pipelines/samplers/visualize.py`
  - diagnostic visualizations, guidance loss plots, point-cloud previews.

## Suggested next cleanup before a larger release

1. Separate MM-TRELLIS-specific functions from the long `trellis_image_to_3d.py` file into smaller modules.
2. Rename experiment-specific function names to public API names, e.g. `run_mm_trellis_guided()`.
3. Add a minimal smoke test that imports the package and validates CLI argument parsing without loading model weights.
4. Add versioned configuration files for the paper's default hyperparameters.
