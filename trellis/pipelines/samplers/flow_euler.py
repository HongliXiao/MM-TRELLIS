from __future__ import annotations

from typing import Any, Optional, Tuple

import os

import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from easydict import EasyDict as edict
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices with Rodrigues' formula.

    Args:
        axis_angle: Tensor with shape (..., 3). A shape of (3,) returns (3, 3);
            a shape of (B, 3) returns (B, 3, 3).

    Returns:
        Rotation matrices with shape (..., 3, 3).

    This local implementation avoids version differences in
    utils3d.torch.transforms.axis_angle_to_matrix.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"axis_angle must have last dimension 3, got shape {tuple(axis_angle.shape)}."
        )

    orig_shape = axis_angle.shape[:-1]
    vec = axis_angle.reshape(-1, 3)

    theta = torch.linalg.norm(vec, dim=-1, keepdim=True)
    theta_safe = torch.clamp(theta, min=1e-8)
    axis = vec / theta_safe

    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)

    skew = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    eye = torch.eye(3, dtype=vec.dtype, device=vec.device).unsqueeze(0)
    eye = eye.expand(vec.shape[0], 3, 3)

    theta = theta.reshape(-1, 1, 1)
    theta_safe_reshaped = theta_safe.reshape(-1, 1, 1)

    # Use Taylor-safe coefficients near zero to keep gradients stable.
    theta2 = theta_safe_reshaped * theta_safe_reshaped
    sin_over_theta = torch.where(
        theta_safe_reshaped < 1e-4,
        1.0 - theta2 / 6.0,
        torch.sin(theta_safe_reshaped) / theta_safe_reshaped,
    )
    one_minus_cos_over_theta2 = torch.where(
        theta_safe_reshaped < 1e-4,
        0.5 - theta2 / 24.0,
        (1.0 - torch.cos(theta_safe_reshaped)) / theta2,
    )

    # Since skew is built from normalized axis, Rodrigues is:
    # R = I + sin(theta) K + (1 - cos(theta)) K^2
    # Equivalently using vec-skew would use the coefficients above. Here we keep
    # normalized-axis form for clarity.
    rotation = eye + torch.sin(theta_safe_reshaped) * skew
    rotation = rotation + (1.0 - torch.cos(theta_safe_reshaped)) * torch.bmm(skew, skew)

    # Exactly zero vector should map to identity.
    zero_mask = (theta.reshape(-1) < 1e-8)
    if zero_mask.any():
        rotation = torch.where(
            zero_mask.view(-1, 1, 1),
            eye,
            rotation,
        )

    return rotation.reshape(*orig_shape, 3, 3)


def voxelize(
    points: np.ndarray,
    device: torch.device,
    output_path: Optional[str] = None,
    resolution: int = 64,
) -> torch.Tensor:
    """Voxelize normalized points in [-0.5, 0.5] into [1, R, R, R]."""
    eps = 1e-6
    points = np.asarray(points, dtype=np.float32)
    points = np.clip(points, -0.5 + eps, 0.5 - eps)

    voxel_indices = ((points + 0.5) * resolution).astype(np.int64)
    voxel_indices = np.clip(voxel_indices, 0, resolution - 1)
    voxel_indices = np.unique(voxel_indices, axis=0)

    voxel_centers = (voxel_indices + 0.5) / resolution - 0.5
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        utils3d.io.write_ply(os.path.join(output_path, f'voxels_in-{len(voxel_centers)}pts.ply'), voxel_centers)

    voxels = torch.zeros(
        1,
        resolution,
        resolution,
        resolution,
        dtype=torch.float32,
        device=device,
    )
    voxels[0, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    return voxels


def normalize_points_torch(points: torch.Tensor, center: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a point set by its maximum side length."""
    min_vals = points.amin(dim=0, keepdim=True)
    max_vals = points.amax(dim=0, keepdim=True)
    scale = 1.0 / torch.clamp((max_vals - min_vals).amax(), min=eps)

    if center:
        points = points - 0.5 * (min_vals + max_vals)
    return points * (scale - eps)


def rotate_z180(points: torch.Tensor) -> torch.Tensor:
    """Rotate points 180 degrees around the world z-axis."""
    rotation = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0], dtype=points.dtype, device=points.device)
    )
    return points @ rotation.T


def get_lr_axis_angle(stage1_cg_params: dict) -> Tuple[float, float, float]:
    lr_axis_angle = stage1_cg_params.get("lr_axis_angle", [0.001, 0.001, 0.5])
    if not isinstance(lr_axis_angle, (list, tuple)) or len(lr_axis_angle) != 3:
        raise ValueError(
            "stage1_cg_params['lr_axis_angle'] must be a list/tuple of three floats, "
            f"got {lr_axis_angle!r}."
        )
    return float(lr_axis_angle[0]), float(lr_axis_angle[1]), float(lr_axis_angle[2])


def rotation_optimization(
    out_coords_normed: torch.Tensor,
    guided_pts: torch.Tensor,
    stage1_cg_params: dict,
    output_path: Optional[str] = None,
) -> torch.Tensor:
    """Align point-cloud guidance to the generated sparse voxel prior.

    Two candidates are optimized:
      1. the original guidance point cloud;
      2. the same point cloud rotated by 180 degrees around the z-axis.

    The candidate with the lower one-way Chamfer distance is returned.
    """
    del output_path  # kept for backward-compatible call signature

    out_coords_normed = normalize_points_torch(out_coords_normed)
    lr_x, lr_y, lr_z = get_lr_axis_angle(stage1_cg_params)

    guided_pts1 = guided_pts
    guided_pts2 = rotate_z180(guided_pts)

    axis_angle_x1 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))
    axis_angle_y1 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))
    axis_angle_z1 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))

    axis_angle_x2 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))
    axis_angle_y2 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))
    axis_angle_z2 = torch.nn.Parameter(torch.zeros(1, device=out_coords_normed.device))

    optimizer_rotation1 = torch.optim.Adam(
        [
            {"params": [axis_angle_x1], "lr": lr_x},
            {"params": [axis_angle_y1], "lr": lr_y},
            {"params": [axis_angle_z1], "lr": lr_z},
        ]
    )
    optimizer_rotation2 = torch.optim.Adam(
        [
            {"params": [axis_angle_x2], "lr": lr_x},
            {"params": [axis_angle_y2], "lr": lr_y},
            {"params": [axis_angle_z2], "lr": lr_z},
        ]
    )

    loss_list1 = []
    loss_list2 = []
    rotated_guided_pts1 = guided_pts1
    rotated_guided_pts2 = guided_pts2

    for _ in range(100):
        optimizer_rotation1.zero_grad()
        optimizer_rotation2.zero_grad()

        axis_angle1 = torch.cat([axis_angle_x1, axis_angle_y1, axis_angle_z1], dim=0)
        axis_angle2 = torch.cat([axis_angle_x2, axis_angle_y2, axis_angle_z2], dim=0)

        # utils3d.torch.transforms.axis_angle_to_matrix expects a standard
        # axis-angle vector with shape (..., 3) in the current utils3d version.
        r1 = axis_angle_to_matrix(axis_angle1.unsqueeze(0))[0]
        r2 = axis_angle_to_matrix(axis_angle2.unsqueeze(0))[0]

        rotated_guided_pts1 = guided_pts1 @ r1.T.float()
        rotated_guided_pts2 = guided_pts2 @ r2.T.float()

        loss1, _ = chamfer_distance(
            rotated_guided_pts1.unsqueeze(0),
            out_coords_normed.unsqueeze(0),
            single_directional=True,
        )
        loss2, _ = chamfer_distance(
            rotated_guided_pts2.unsqueeze(0),
            out_coords_normed.unsqueeze(0),
            single_directional=True,
        )

        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())

        loss1.backward()
        loss2.backward()
        optimizer_rotation1.step()
        optimizer_rotation2.step()

    if loss_list1[-1] <= loss_list2[-1]:
        return rotated_guided_pts1.detach()
    return rotated_guided_pts2.detach()


class FlowEulerSampler(Sampler):
    """Euler sampler for flow-matching models."""

    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t: torch.Tensor, t: float, eps: torch.Tensor) -> torch.Tensor:
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t: torch.Tensor, t: float, x_0: torch.Tensor) -> torch.Tensor:
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t: torch.Tensor, t: float, v: torch.Tensor):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t: torch.Tensor, t: float, cond=None, **kwargs):
        t_tensor = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t_tensor, cond, **kwargs)

    def _get_model_prediction(self, model, x_t: torch.Tensor, t: float, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _add_noise(self, x_0: torch.Tensor, t, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = torch.tensor(t, dtype=torch.float32, device=x_0.device)
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        return (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs,
    ):
        pred_x_0, _, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    def _sample_with_point_guidance(
        self,
        model,
        sample: torch.Tensor,
        cond,
        t_pairs,
        verbose: bool,
        coarse_point_clouds: torch.Tensor,
        ss_decoder,
        output_path: Optional[str],
        stage1_cg_params: dict,
        resolution: int,
        **kwargs,
    ):
        sample = torch.nn.Parameter(sample)
        optimizer = torch.optim.Adam(
            [{"params": [sample], "lr": stage1_cg_params.get("lr_sample", 0.03)}]
        )

        sample_optimization_ratio = 0.7
        sample_optimization_start_step = int(len(t_pairs) * (1 - sample_optimization_ratio)) - 1

        coarse_voxels = None
        sparse_mask = None
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            optimizer.zero_grad()

            if i > sample_optimization_start_step:
                pred_v = self._inference_model(model, sample, t, cond, **kwargs)
                with torch.no_grad():
                    pred_eps = (1 - t) * pred_v + sample

                pred_x_0 = (
                    (1 - self.sigma_min) * sample
                    - (self.sigma_min + (1 - self.sigma_min) * t) * pred_v.detach()
                )
                logits_decoder = ss_decoder(pred_x_0)

                if sparse_mask is None or coarse_voxels is None:
                    raise RuntimeError("Point-cloud guidance mask was not initialized.")

                loss = F.binary_cross_entropy_with_logits(
                    logits_decoder[sparse_mask],
                    coarse_voxels[sparse_mask],
                )
                loss = loss * stage1_cg_params.get("weight", 0.01)
                loss.backward()

                with torch.no_grad():
                    pred_epsilon_norm = torch.linalg.norm(pred_eps).item()
                    latent_grad_norm = torch.linalg.norm(sample.grad).item()
                    sample.grad *= pred_epsilon_norm / max(latent_grad_norm, 1e-8)

                optimizer.step()

                if stage1_cg_params.get("update_v", False) and i < len(t_pairs) - 1:
                    with torch.no_grad():
                        pred_v = self._inference_model(
                            model,
                            sample,
                            t,
                            cond,
                            repeat_cond=True,
                            **kwargs,
                        )
            else:
                with torch.no_grad():
                    pred_x_0, _, pred_v = self._get_model_prediction(
                        model,
                        sample,
                        t,
                        cond,
                        **({**(kwargs or {}), "cfg_strength": 5.0}),
                    )
                    logits_decoder = ss_decoder(pred_x_0)

                if i == sample_optimization_start_step:
                    stage1_coords = torch.argwhere(logits_decoder > 0)[:, [0, 2, 3, 4]].int()
                    if stage1_coords.numel() == 0:
                        final_guided_pts = coarse_point_clouds
                    else:
                        out_coords_normed = (stage1_coords[:, 1:4] + 0.5) / resolution - 0.5
                        final_guided_pts = rotation_optimization(
                            out_coords_normed=out_coords_normed,
                            guided_pts=coarse_point_clouds,
                            stage1_cg_params=stage1_cg_params,
                            output_path=output_path,
                        )

                    coarse_voxels = voxelize(
                        final_guided_pts.detach().cpu().numpy(),
                        device=sample.device,
                        output_path=None,
                        resolution=resolution,
                    ).unsqueeze(1)
                    sparse_mask = coarse_voxels > 0

            with torch.no_grad():
                pred_x_prev = sample - (t - t_prev) * pred_v
                out = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
                sample.data = pred_x_prev

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret

    def sample(
        self,
        model,
        noise: torch.Tensor,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        coarse_point_clouds: Optional[torch.Tensor] = None,
        coarse_latent=None,
        denoising_strength=None,
        coarse_voxels=None,
        ss_encoder=None,
        ss_decoder=None,
        output_path: Optional[str] = None,
        stage1_cg_params: Optional[dict] = None,
        resolution: int = 64,
        **kwargs,
    ):
        del coarse_voxels, ss_encoder  # retained in signature for compatibility

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        if (denoising_strength is not None) and (coarse_latent is not None):
            start_index = min(int(round(len(t_seq) * (1 - denoising_strength))), len(t_seq) - 1)
            t_seq = t_seq[start_index:]
            sample = self._add_noise(coarse_latent, t_seq[0:1], noise=noise)

        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(len(t_seq) - 1)]

        if (ss_decoder is not None) and (coarse_point_clouds is not None):
            return self._sample_with_point_guidance(
                model=model,
                sample=sample,
                cond=cond,
                t_pairs=t_pairs,
                verbose=verbose,
                coarse_point_clouds=coarse_point_clouds,
                ss_decoder=ss_decoder,
                output_path=output_path,
                stage1_cg_params=stage1_cg_params or {},
                resolution=resolution,
                **kwargs,
            )

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """Euler sampler with classifier-free guidance."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            **kwargs,
        )


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """Euler sampler with classifier-free guidance applied in a time interval."""

    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )
