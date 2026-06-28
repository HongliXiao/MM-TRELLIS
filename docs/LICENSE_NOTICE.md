# License notice

This repository contains code derived from or compatible with the original TRELLIS project and also includes third-party components embedded in the `trellis/` tree. Several files already carry explicit upstream headers and restrictions.

Before public release, please review at least the following:

- original TRELLIS repository license;
- 3D Gaussian Splatting / Inria components under `trellis/renderers/` and `trellis/representations/gaussian/`;
- NVIDIA FlexiCubes components under `trellis/representations/mesh/flexicubes/`;
- external runtime dependencies such as Kaolin, nvdiffrast, diff-gaussian-rasterization, spconv, xFormers, PyTorch3D, Open3D, and utils3d;
- example data licensing, especially if any samples are derived from the Waymo Open Dataset.

Do not add a permissive top-level license such as MIT/Apache-2.0 unless you have verified that all included code and data are compatible with that choice. If the release is intended for research-only use, state this explicitly in the README and license file.

Suggested attribution for Waymo-derived example data, when applicable:

> This example data was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at https://waymo.com/open/terms, and your access and use of such work are governed by the terms and conditions therein.
