from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Literal, Optional

import numpy as np
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import samplers
from .base import Pipeline
from ..modules import sparse as sp
from ..representations.gaussian.general_utils import inverse_sigmoid


def rotate_points(points: torch.Tensor, angles, degrees: bool = True) -> torch.Tensor:
    """Rotate 3D points by Euler angles (x, y, z)."""
    import math

    if degrees:
        angles = [math.radians(a) for a in angles]
    ax, ay, az = angles

    rx = torch.tensor(
        [[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]],
        dtype=torch.float32,
        device=points.device,
    )
    ry = torch.tensor(
        [[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]],
        dtype=torch.float32,
        device=points.device,
    )
    rz = torch.tensor(
        [[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]],
        dtype=torch.float32,
        device=points.device,
    )
    rotation = rz @ ry @ rx
    return points @ rotation.T


def intersect_coords(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    resolution: int = 64,
    return_mask: bool = True,
):
    """Intersect sparse voxel coordinates and optionally return the mask on coords1."""
    c1 = coords1[:, 1:4]
    c2 = coords2[:, 1:4] if coords2.shape[1] == 4 else coords2

    ids1 = c1[:, 0] * (resolution**2) + c1[:, 1] * resolution + c1[:, 2]
    ids2 = c2[:, 0] * (resolution**2) + c2[:, 1] * resolution + c2[:, 2]

    mask = torch.isin(ids1, ids2)
    common_ids = ids1[mask]

    x = common_ids // (resolution**2)
    y = (common_ids % (resolution**2)) // resolution
    z = common_ids % resolution
    batch = torch.zeros_like(x)
    inter_coords = torch.stack([batch, x, y, z], dim=1)

    if return_mask:
        return inter_coords, mask
    return inter_coords


class TrellisImageTo3DPipeline(Pipeline):
    """Minimal MM-TRELLIS image-to-3D pipeline.

    This cleaned version keeps only the components required by
    ``run_multi_image_return_ss_with_grad_and_filter_gs_pts()``.
    """

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        sparse_structure_sampler: Optional[samplers.Sampler] = None,
        slat_sampler: Optional[samplers.Sampler] = None,
        slat_normalization: Optional[dict] = None,
        image_cond_model: Optional[str] = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]
        new_pipeline._init_image_cond_model(args["image_cond_model"])
        return new_pipeline

    def _init_image_cond_model(self, name: str):
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        self.image_cond_model_transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Use input alpha if available; otherwise remove background and crop."""
        has_alpha = False
        if image.mode == "RGBA":
            alpha = np.array(image)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        if has_alpha:
            output = image
        else:
            image = image.convert("RGB")
            max_size = max(image.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            if getattr(self, "rembg_session", None) is None:
                self.rembg_session = rembg.new_session("u2net")
            output = rembg.remove(image, session=self.rembg_session)

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255.0
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)."
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should contain PIL images."
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255.0 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        return F.layer_norm(features, features.shape[-1:])

    def get_cond(self, image) -> Dict[str, torch.Tensor]:
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {"cond": cond, "neg_cond": neg_cond}

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: Optional[dict] = None,
        denoising_strength=None,
        coarse_point_clouds=None,
        coarse_voxels=None,
        output_path=None,
        stage1_cg_params=None,
    ) -> torch.Tensor:
        sampler_params = sampler_params or {}

        flow_model = self.models["sparse_structure_flow_model"]
        decoder = self.models["sparse_structure_decoder"]
        encoder = self.models["sparse_structure_encoder"]
        reso = flow_model.resolution

        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        if (denoising_strength is not None) and (coarse_voxels is not None):
            coarse_latent = encoder(coarse_voxels.unsqueeze(1), sample_posterior=False)
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                coarse_latent=coarse_latent,
                denoising_strength=denoising_strength,
            ).samples
        elif (coarse_point_clouds is not None) and (coarse_voxels is not None):
            torch.set_grad_enabled(True)
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                coarse_voxels=coarse_voxels.unsqueeze(1),
                coarse_point_clouds=coarse_point_clouds,
                ss_encoder=encoder,
                ss_decoder=decoder,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params,
            ).samples
            torch.set_grad_enabled(False)
        elif coarse_voxels is not None:
            torch.set_grad_enabled(True)
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                coarse_voxels=coarse_voxels.unsqueeze(1),
                ss_encoder=encoder,
                ss_decoder=decoder,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params,
            ).samples
            torch.set_grad_enabled(False)
        elif coarse_point_clouds is not None:
            torch.set_grad_enabled(True)
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                coarse_point_clouds=coarse_point_clouds,
                ss_encoder=encoder,
                ss_decoder=decoder,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params,
            ).samples
            torch.set_grad_enabled(False)
        else:
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                ss_decoder=decoder,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params,
            ).samples

        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: Optional[dict] = None,
        denoising_strength=None,
        coarse_latent=None,
    ) -> sp.SparseTensor:
        sampler_params = sampler_params or {}

        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        if (denoising_strength is not None) and (coarse_latent is not None):
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                coarse_latent=coarse_latent,
                denoising_strength=denoising_strength,
            ).samples
        else:
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
            ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        return slat * std + mean

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: Optional[List[str]] = None,
    ) -> dict:
        formats = formats or ["mesh", "gaussian", "radiance_field"]
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """Temporarily patch the sampler so it can condition on multiple images."""
        sampler = getattr(self, sampler_name)
        setattr(sampler, "_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, repeat_cond=False, **kwargs):
                if not repeat_cond:
                    cond_idx = cond_indices.pop(0)
                    cond_i = cond[cond_idx : cond_idx + 1]
                else:
                    cond_idx = (cond_indices[0] - 1) % num_images
                    cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                preds = [
                    FlowEulerSampler._inference_model(self, model, x_t, t, cond[i : i + 1], **kwargs)
                    for i in range(len(cond))
                ]
                pred = sum(preds) / len(preds)
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        yield
        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, "_old_inference_model")

    def run_multi_image_return_ss_with_grad_and_filter_gs_pts(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        formats: Optional[List[str]] = None,
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        denoising_strength=None,
        coarse_point_clouds=None,
        coarse_voxels=None,
        output_path=None,
        stage1_cg_params=None,
    ):
        """MM-TRELLIS multi-view inference with Gaussian-opacity filtering.

        This is the only public inference entry retained in the cleaned pipeline.
        """
        sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        slat_sampler_params = slat_sampler_params or {}
        formats = formats or ["mesh", "gaussian"]

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]

        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)

        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps")
        with self.inject_sampler_multi_image("sparse_structure_sampler", len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(
                cond,
                num_samples,
                sparse_structure_sampler_params,
                denoising_strength=denoising_strength,
                coarse_point_clouds=coarse_point_clouds,
                coarse_voxels=coarse_voxels,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params,
            )
        if coords.numel() == 0:
            return coords, None

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image("slat_sampler", len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)

        threshold = -5.0
        resolution = 64
        gs = self.models["slat_decoder_gs"](slat)[0]
        xyz = gs.get_xyz.detach()
        opacity_field = inverse_sigmoid(gs.get_opacity).detach().squeeze(1)

        filtered_pts = xyz[opacity_field >= threshold]
        rotated_pts_filtered = rotate_points(filtered_pts, (0, 0, 0))
        filtered_coords = ((rotated_pts_filtered + 0.5) * resolution).int()
        filtered_unique_coords = torch.from_numpy(np.unique(filtered_coords.cpu().numpy(), axis=0)).to(coords.device)

        _, mask = intersect_coords(coords, filtered_unique_coords, resolution)
        slat_filtered = sp.SparseTensor(feats=slat.feats[mask], coords=slat.coords[mask])
        return coords, self.decode_slat(slat_filtered, formats)
