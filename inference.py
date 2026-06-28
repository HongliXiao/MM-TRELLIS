"""MM-TRELLIS multi-view inference entry point.

This script runs multi-image conditioned TRELLIS generation with LiDAR/point-cloud
sparse-structure guidance and opacity-based voxel filtering.  For each instance,
all images under ``<input_root>/<instance>/<split>/`` are loaded as one multi-view
input and passed to ``run_multi_image_return_ss_with_grad_and_filter_gs_pts``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

# Keep these environment defaults before importing TRELLIS modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
os.environ.setdefault("ATTN_BACKEND", "xformers")  # xformers or flash-attn
# os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
import numpy as np
import open3d as o3d
import torch
import utils3d
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


def coords_to_positions(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Convert sparse voxel coordinates [N, 4] to positions in [-0.5, 0.5]."""
    indices = coords[:, 1:4].float()
    return indices / resolution - 0.5


def normalize_points(points: np.ndarray, mode: str = "scale-only", scale: float = 0.98) -> np.ndarray:
    """Normalize point cloud coordinates for TRELLIS point-cloud guidance.

    Args:
        points: Input point cloud with shape [N, 3].
        mode: ``scale-only`` scales the point cloud by its maximum side length
            while preserving the original coordinate center. This matches the
            experiment script that used ``pts * (0.98 / max_span)``. ``unit-box``
            recenters the point cloud and scales it into a unit cube. ``none``
            keeps the input coordinates unchanged.
        scale: Target maximum side length after normalization.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N, 3] point cloud, got shape {points.shape}.")

    if mode == "none":
        return points

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    max_span = float(np.max(max_vals - min_vals))
    if max_span <= 0:
        raise ValueError("Degenerate point cloud: all points are identical.")

    scale_factor = scale / max_span
    if mode == "scale-only":
        return points * scale_factor
    if mode == "unit-box":
        center = (min_vals + max_vals) / 2.0
        return (points - center) * scale_factor
    raise ValueError(f"Unknown point normalization mode: {mode}")


def list_instances(input_root: Path, split: str, requested: Sequence[str] | None) -> list[str]:
    if requested:
        return list(requested)
    instances = [p.name for p in sorted(input_root.iterdir()) if (p / split).is_dir()]
    if not instances:
        raise FileNotFoundError(f"No instances found under {input_root} with split={split!r}.")
    return instances


def iter_image_paths(
    instance_dir: Path,
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp"),
    max_views: int | None = None,
) -> list[Path]:
    """Return sorted image paths for one instance/split directory."""
    valid_exts = {e.lower() for e in exts}
    image_paths = [p for p in sorted(instance_dir.iterdir()) if p.suffix.lower() in valid_exts]
    if max_views is not None:
        image_paths = image_paths[:max_views]
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {instance_dir}.")
    return image_paths


def load_multi_view_images(image_paths: Sequence[Path], image_mode: str = "unchanged") -> list[Image.Image]:
    """Load all views as PIL images.

    By default, the original image mode is preserved. This is important when the
    input PNGs contain alpha masks; forcing RGB may discard useful foreground
    information before TRELLIS preprocessing.
    """
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            if image_mode != "unchanged":
                image = image.convert(image_mode)
            images.append(image.copy())
    return images


def default_point_cloud_path(pts_root: Path, instance: str, filename: str) -> Path:
    """Resolve the default DriveStudio/Waymo-style point-cloud path.

    Expected example:
    examples_input/drivestudio_way_processed/006/static_006_002/
        aggregated_instance_lidar_pts/2.symmetric-DBSCAN.ply
    """
    parts = instance.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot infer scene id from instance name {instance!r}. "
            "Use the static_<scene>_<track> naming convention or edit this resolver."
        )
    scene_name = parts[1]
    return pts_root / scene_name / instance / "aggregated_instance_lidar_pts" / filename


def load_guidance_points(
    path: Path,
    normalization: str,
    normalization_scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Point-cloud file not found: {path}")
    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {path}")
    points = normalize_points(points, mode=normalization, scale=normalization_scale)
    return torch.as_tensor(points, dtype=torch.float32, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MM-TRELLIS multi-view point-cloud guided inference.")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("examples_input/initial_test_instances"),
        help="Root directory containing instance folders and image splits.",
    )
    parser.add_argument(
        "--pts_root",
        type=Path,
        default=Path("examples_input/drivestudio_way_processed"),
        help="Root directory containing preprocessed instance LiDAR point clouds.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/mm_trellis"),
        help="Directory where generated assets and videos are saved.",
    )
    parser.add_argument(
        "--pretrained_model",
        default="microsoft/TRELLIS-image-large",
        help="Hugging Face repo id or local path for the TRELLIS image model.",
    )
    parser.add_argument("--split", default="test", help="Image split inside each instance folder, e.g. test or validation.")
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Instance ids to process. If omitted, all instances under --input_root are used.",
    )
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument(
        "--point_cloud_name",
        default="2.symmetric-DBSCAN.ply",
        help="Point-cloud filename inside aggregated_instance_lidar_pts/.",
    )
    parser.add_argument(
        "--point_normalization",
        choices=["scale-only", "unit-box", "none"],
        default="scale-only",
        help="Point-cloud normalization mode. scale-only matches the original experiment script.",
    )
    parser.add_argument(
        "--point_normalization_scale",
        type=float,
        default=0.98,
        help="Maximum side length after point-cloud normalization.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device. CUDA is required for practical inference.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG strength for sparse-structure sampling.")
    parser.add_argument(
        "--lr_sample",
        type=float,
        default=0.03,
        help="Latent sample learning rate.",
    )
    parser.add_argument(
        "--lr_axis_angle",
        type=float,
        nargs=3,
        default=[0.001, 0.001, 0.5],
    )
    parser.add_argument("--weight", type=float, default=0.01, help="Guidance loss weight.")
    parser.add_argument("--step", type=int, default=100, help="Sparse-structure denoising steps.")
    parser.add_argument("--update_v", action="store_true", help="Update flow velocity during guided sampling.")
    parser.add_argument(
        "--chamfer_distance_single_directional",
        action="store_true",
        default=True,
        help="Use single-directional Chamfer distance for point-cloud guidance.",
    )
    parser.add_argument("--min_views", type=int, default=2, help="Minimum number of views required for multi-view inference.")
    parser.add_argument("--max_views", type=int, default=None, help="Optional maximum number of sorted views to use.")
    parser.add_argument(
        "--image_mode",
        choices=["unchanged", "RGB", "RGBA"],
        default="unchanged",
        help="PIL image mode conversion. Keep unchanged by default to preserve alpha masks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if the final PLY exists.")
    parser.add_argument("--save_video", dest="save_video", action="store_false", default=True, help="Do not save rendered MP4 video.")
    parser.add_argument("--save_glb", dest="save_glb", action="store_false", default=True, help="Do not save GLB mesh output.")
    parser.add_argument("--save_gaussian_ply", dest="save_gaussian_ply", action="store_false", default=True, help="Do not save Gaussian PLY output.")
    parser.add_argument("--save_voxels", dest="save_voxels", action="store_false", default=True, help="Do not save sparse voxel PLY output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TRELLIS model from {args.pretrained_model}...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pretrained_model)
    if device.type == "cuda":
        pipeline.cuda()
    elif hasattr(pipeline, "to"):
        pipeline.to(device)
    print("Model loaded.")

    instances = list_instances(args.input_root, args.split, args.instances)
    end_id = None if args.end_id is None or args.end_id < 0 else args.end_id
    instances = instances[args.start_id:end_id]
    if not instances:
        raise ValueError("No instances left to process after applying --start_id/--end_id.")

    for instance in instances:
        input_dir = args.input_root / instance / args.split
        point_cloud_path = default_point_cloud_path(args.pts_root, instance, args.point_cloud_name)
        instance_output_dir = args.output_dir / instance
        instance_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            guidance_points = load_guidance_points(
                point_cloud_path,
                args.point_normalization,
                args.point_normalization_scale,
                device,
            )
            image_paths = iter_image_paths(input_dir, max_views=args.max_views)
            if len(image_paths) < args.min_views:
                raise ValueError(
                    f"Need at least {args.min_views} views for multi-view inference, "
                    f"but found {len(image_paths)} in {input_dir}."
                )
            images = load_multi_view_images(image_paths, image_mode=args.image_mode)
        except Exception as exc:
            print(f"[SKIP] {instance}: {exc}")
            continue

        print(
            f"[{instance}] multi-view input: {len(images)} images; "
            f"guidance points: {guidance_points.shape[0]}; point cloud: {point_cloud_path}"
        )
        print("  views:")
        for image_path in image_paths:
            print(f"    - {image_path.name}")

        out_name = "output"
        final_ply = instance_output_dir / f"{out_name}.ply"
        if final_ply.exists() and not args.overwrite:
            print(f"[SKIP] {instance}: output already exists")
            continue

        stage1_cg_params = {
            "lr_sample": args.lr_sample,
            "lr_axis_angle": args.lr_axis_angle,
            "weight": args.weight,
            "out_name": out_name,
            "update_v": args.update_v,
            "chamfer_distance_single_directional": args.chamfer_distance_single_directional,
        }

        try:
            coords, outputs = pipeline.run_multi_image_return_ss_with_grad_and_filter_gs_pts(
                images,
                seed=args.seed,
                coarse_point_clouds=guidance_points,
                output_path=str(instance_output_dir),
                sparse_structure_sampler_params={
                    "steps": args.step,
                    "cfg_strength": args.cfg,
                    "cfg_interval": [0.0, 1.0],
                },
                stage1_cg_params=stage1_cg_params,
            )

            if args.save_voxels:
                positions = coords_to_positions(coords, resolution=64)
                utils3d.io.write_ply(
                    str(instance_output_dir / f"{out_name}-voxels_out-{len(coords)}pts.ply"),
                    positions.detach().cpu().numpy(),
                )

            if args.save_video:
                video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
                video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
                video = [
                    np.concatenate([frame_gs, frame_mesh], axis=1)
                    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
                ]
                imageio.mimsave(instance_output_dir / f"{out_name}.mp4", video, fps=30)
                # imageio.mimsave(str(instance_output_dir / f"{out_name}.mp4"), video, fps=30, format="FFMPEG")

            if args.save_glb:
                previous_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                try:
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=0.95,
                        texture_size=1024,
                    )
                finally:
                    torch.set_grad_enabled(previous_grad_state)
                glb.export(instance_output_dir / f"{out_name}.glb")

            if args.save_gaussian_ply:
                outputs["gaussian"][0].save_ply(str(final_ply))

            print(f"[DONE] {instance}: {out_name}")
        except Exception as exc:
            print(f"[ERROR] {instance}: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
