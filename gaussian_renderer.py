import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,  # (N, 3)
        covs3d: torch.Tensor,   # (N, 3, 3)
        K: torch.Tensor,        # (3, 3)
        R: torch.Tensor,        # (3, 3)
        t: torch.Tensor         # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        N = means3D.shape[0]

        # 1. World -> camera
        cam_points = means3D @ R.T + t.unsqueeze(0)  # (N, 3)

        # 2. Depths
        depths_raw = cam_points[:, 2]
        depths = depths_raw.clamp(min=1e-4)

        # 3. Perspective projection
        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[..., 2:3].clamp(min=1e-4)

        # 4. Jacobian of projection:
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        fx = K[0, 0]
        fy = K[1, 1]

        x = cam_points[:, 0]
        y = cam_points[:, 1]
        z = depths

        J_proj = torch.zeros(
            (N, 2, 3),
            device=means3D.device,
            dtype=means3D.dtype
        )

        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / (z * z)

        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / (z * z)

        # 5. Transform covariance from world space to camera space
        R_expand = R.unsqueeze(0).expand(N, -1, -1)
        covs_cam = torch.bmm(
            R_expand,
            torch.bmm(covs3d, R_expand.transpose(1, 2))
        )

        # 6. Project 3D covariance to 2D
        covs2D = torch.bmm(
            J_proj,
            torch.bmm(covs_cam, J_proj.transpose(1, 2))
        )

        return means2D, covs2D, depths_raw

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,  # (N, 2)
        covs2D: torch.Tensor,   # (N, 2, 2)
        pixels: torch.Tensor    # (H, W, 2)
    ) -> torch.Tensor:
        # Return: (N, H, W)

        N = means2D.shape[0]
        H, W = pixels.shape[:2]

        # dx: (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)

        # Numerical stability
        eps = 1e-4
        eye = torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)
        covs2D = covs2D + eps * eye

        # determinant
        det = (
            covs2D[:, 0, 0] * covs2D[:, 1, 1]
            - covs2D[:, 0, 1] * covs2D[:, 1, 0]
        )
        det = det.clamp(min=1e-8)

        # inverse covariance
        inv_covs = torch.linalg.inv(covs2D)  # (N, 2, 2)

        # Mahalanobis distance:
        # dx^T Sigma^{-1} dx
        mahal = torch.einsum(
            "nhwi,nij,nhwj->nhw",
            dx,
            inv_covs,
            dx
        )

        # Gaussian PDF
        norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det))
        norm = norm.view(N, 1, 1)

        gaussian = norm * torch.exp(-0.5 * mahal)

        # Avoid rare numerical explosion
        gaussian = torch.nan_to_num(gaussian, nan=0.0, posinf=0.0, neginf=0.0)

        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = depths > 1e-4
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # Clamp alpha for stable front-to-back compositing
        alphas = alphas.clamp(min=0.0, max=0.99)  # (N, H, W)

        # Front-to-back transmittance:
        # T_i = prod_{j < i} (1 - alpha_j)
        ones = torch.ones(
            (1, self.H, self.W),
            device=alphas.device,
            dtype=alphas.dtype
        )

        transmittance = torch.cumprod(
            torch.cat([ones, 1.0 - alphas + 1e-10], dim=0),
            dim=0
        )[:-1]

        weights = alphas * transmittance  # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
