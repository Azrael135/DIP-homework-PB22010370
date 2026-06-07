import torch
import torch.nn as nn
from typing import Tuple, Optional


class GaussianRenderer(nn.Module):
    """A simple differentiable Gaussian renderer with optional runtime checks.

    Usage for debugging blank renders:
        renderer = GaussianRenderer(H, W, debug=True, debug_every=1)

    Important switches:
        debug=True
            Print shape/range/finite diagnostics during forward().
        strict_checks=True
            Raise an error on invalid tensors instead of only printing warnings.
        use_pdf_normalization=False
            3DGS-style unnormalized Gaussian kernel. Set True to reproduce the
            original PDF-normalized behavior from your pasted code.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        debug: bool = False,
        debug_every: int = 1,
        strict_checks: bool = False,
        use_pdf_normalization: bool = False,
        tiny_threshold: float = 1e-8,
    ):
        super().__init__()
        self.H = int(image_height)
        self.W = int(image_width)
        self.debug = bool(debug)
        self.debug_every = max(int(debug_every), 1)
        self.strict_checks = bool(strict_checks)
        self.use_pdf_normalization = bool(use_pdf_normalization)
        self.tiny_threshold = float(tiny_threshold)
        self._forward_count = 0

        # Pre-compute pixel coordinates grid.
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        # Shape: (H, W, 2), pixel[..., 0] = x, pixel[..., 1] = y.
        self.register_buffer("pixels", torch.stack([x, y], dim=-1))

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _should_debug(self) -> bool:
        return self.debug and (self._forward_count % self.debug_every == 0)

    def _warn_or_raise(self, msg: str) -> None:
        msg = f"[GaussianRenderer check] {msg}"
        if self.strict_checks:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    @staticmethod
    def _shape_matches(actual: Tuple[int, ...], expected: Tuple[Optional[int], ...]) -> bool:
        if len(actual) != len(expected):
            return False
        return all(e is None or a == e for a, e in zip(actual, expected))

    def _check_shape(self, name: str, tensor: torch.Tensor, expected: Tuple[Optional[int], ...]) -> None:
        actual = tuple(tensor.shape)
        if not self._shape_matches(actual, expected):
            self._warn_or_raise(f"{name} shape is {actual}, expected {expected}.")

    def _check_finite(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            self._warn_or_raise(f"{name} is empty.")
            return
        finite = torch.isfinite(tensor)
        if not bool(finite.all().item()):
            bad = tensor.numel() - int(finite.sum().item())
            self._warn_or_raise(f"{name} contains {bad}/{tensor.numel()} non-finite values.")

    def _stats(self, name: str, tensor: torch.Tensor) -> str:
        with torch.no_grad():
            x = tensor.detach()
            if x.numel() == 0:
                return f"{name}: empty"

            finite = torch.isfinite(x)
            finite_count = int(finite.sum().item())
            total = x.numel()
            if finite_count == 0:
                return f"{name}: shape={tuple(x.shape)}, finite=0/{total}"

            xf = x[finite]
            nonzero_ratio = float((xf.abs() > self.tiny_threshold).float().mean().item())
            return (
                f"{name}: shape={tuple(x.shape)}, "
                f"min={xf.min().item():.6g}, max={xf.max().item():.6g}, "
                f"mean={xf.mean().item():.6g}, finite={finite_count}/{total}, "
                f"|x|>{self.tiny_threshold:g} ratio={nonzero_ratio:.4f}"
            )

    def _print_debug_block(self, title: str, lines) -> None:
        if not self._should_debug():
            return
        print(f"\n========== GaussianRenderer Debug: {title} | forward #{self._forward_count} ==========")
        for line in lines:
            print(line)
        print("====================================================================\n")

    # ------------------------------------------------------------------
    # Core rendering functions
    # ------------------------------------------------------------------
    def compute_projection(
        self,
        means3D: torch.Tensor,  # (N, 3)
        covs3d: torch.Tensor,   # (N, 3, 3)
        K: torch.Tensor,        # (3, 3)
        R: torch.Tensor,        # (3, 3)
        t: torch.Tensor,        # (3,) or (3, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        t = t.reshape(3)

        # 1. World -> camera.
        cam_points = means3D @ R.T + t.unsqueeze(0)  # (N, 3)

        # 2. Depths. Keep raw depths for masking; use safe depths only for division.
        depths_raw = cam_points[:, 2]
        depths_safe = depths_raw.clamp(min=1e-4)

        # 3. Perspective projection.
        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[..., 2:3].clamp(min=1e-4)

        # 4. Jacobian of projection:
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        fx = K[0, 0]
        fy = K[1, 1]

        x = cam_points[:, 0]
        y = cam_points[:, 1]
        z = depths_safe

        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / (z * z)
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / (z * z)

        # 5. Transform covariance from world space to camera space.
        R_expand = R.unsqueeze(0).expand(N, -1, -1)
        covs_cam = torch.bmm(R_expand, torch.bmm(covs3d, R_expand.transpose(1, 2)))

        # 6. Project 3D covariance to 2D.
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.transpose(1, 2)))

        # Symmetrize to remove tiny numerical asymmetry.
        covs2D = 0.5 * (covs2D + covs2D.transpose(1, 2))

        return means2D, covs2D, depths_raw

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,  # (N, 2)
        covs2D: torch.Tensor,   # (N, 2, 2)
        pixels: torch.Tensor,   # (H, W, 2)
    ) -> torch.Tensor:
        # Return: (N, H, W)
        N = means2D.shape[0]

        # dx: (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)

        # Numerical stability.
        eps = 1e-4
        eye = torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)
        covs2D = covs2D + eps * eye
        covs2D = 0.5 * (covs2D + covs2D.transpose(1, 2))

        # determinant before clamp, for debugging invalid projected covariance.
        det_raw = covs2D[:, 0, 0] * covs2D[:, 1, 1] - covs2D[:, 0, 1] * covs2D[:, 1, 0]
        if self._should_debug():
            num_bad_det = int((det_raw <= 0).sum().item())
            if num_bad_det > 0:
                self._warn_or_raise(
                    f"covs2D has {num_bad_det}/{N} non-positive determinants before clamp. "
                    "This often means projected covariance is invalid or scale is too small."
                )

        det = det_raw.clamp(min=1e-8)

        # Inverse covariance.
        try:
            inv_covs = torch.linalg.inv(covs2D)
        except RuntimeError as e:
            self._warn_or_raise(f"torch.linalg.inv(covs2D) failed; falling back to pinv. Original error: {e}")
            inv_covs = torch.linalg.pinv(covs2D)

        # Mahalanobis distance: dx^T Sigma^{-1} dx.
        mahal = torch.einsum("nhwi,nij,nhwj->nhw", dx, inv_covs, dx)
        mahal = torch.clamp(mahal, min=0.0, max=1e6)

        # For 3DGS-style splatting, the unnormalized kernel is usually better:
        # center value = 1. The PDF normalization can make values extremely tiny.
        gaussian = torch.exp(-0.5 * mahal)
        if self.use_pdf_normalization:
            norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det))
            gaussian = norm.view(N, 1, 1) * gaussian

        gaussian = torch.nan_to_num(gaussian, nan=0.0, posinf=0.0, neginf=0.0)
        return gaussian

    def forward(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        colors: torch.Tensor,           # (N, 3)
        opacities: torch.Tensor,        # (N, 1) or (N,)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor,                # (3,) or (3, 1)
    ) -> torch.Tensor:
        self._forward_count += 1
        N = means3D.shape[0]
        device = means3D.device
        dtype = means3D.dtype

        # Make camera tensors and pixel buffer consistent with Gaussian tensors.
        K = K.reshape(3, 3).to(device=device, dtype=dtype)
        R = R.reshape(3, 3).to(device=device, dtype=dtype)
        t = t.reshape(3).to(device=device, dtype=dtype)
        pixels = self.pixels.to(device=device, dtype=dtype)

        if opacities.ndim == 1:
            opacities = opacities[:, None]

        # Basic input checks.
        self._check_shape("means3D", means3D, (None, 3))
        self._check_shape("covs3d", covs3d, (N, 3, 3))
        self._check_shape("colors", colors, (N, 3))
        self._check_shape("opacities", opacities, (N, 1))
        self._check_shape("K", K, (3, 3))
        self._check_shape("R", R, (3, 3))
        self._check_shape("t", t, (3,))

        for name, tensor in [
            ("means3D", means3D),
            ("covs3d", covs3d),
            ("colors", colors),
            ("opacities", opacities),
            ("K", K),
            ("R", R),
            ("t", t),
        ]:
            self._check_finite(name, tensor)

        if self._should_debug():
            self._print_debug_block(
                "inputs",
                [
                    self._stats("means3D", means3D),
                    self._stats("covs3d", covs3d),
                    self._stats("colors", colors),
                    self._stats("opacities", opacities),
                    self._stats("K", K),
                    self._stats("R", R),
                    self._stats("t", t),
                    f"use_pdf_normalization={self.use_pdf_normalization}",
                ],
            )

        # 1. Project to 2D.
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        self._check_finite("means2D", means2D)
        self._check_finite("covs2D", covs2D)
        self._check_finite("depths", depths)

        # 2. Depth mask.
        valid_depth_mask = depths > 1e-4

        # Extra projection diagnostics: are points in front of camera and inside image?
        in_image_mask = (
            (means2D[:, 0] >= 0) & (means2D[:, 0] <= self.W - 1) &
            (means2D[:, 1] >= 0) & (means2D[:, 1] <= self.H - 1)
        )
        visible_center_mask = valid_depth_mask & in_image_mask

        if self._should_debug():
            valid_depth_count = int(valid_depth_mask.sum().item())
            in_image_count = int(in_image_mask.sum().item())
            visible_center_count = int(visible_center_mask.sum().item())
            if valid_depth_count == 0:
                self._warn_or_raise("No Gaussian has positive depth. Check R/t convention or depth scale.")
            if visible_center_count == 0:
                self._warn_or_raise(
                    "No Gaussian center is both in front of the camera and inside the image. "
                    "If rendered is blank, check K/R/t, image size, or COLMAP coordinate convention."
                )

            det2d = covs2D[:, 0, 0] * covs2D[:, 1, 1] - covs2D[:, 0, 1] * covs2D[:, 1, 0]
            self._print_debug_block(
                "projection",
                [
                    self._stats("means2D_x", means2D[:, 0]),
                    self._stats("means2D_y", means2D[:, 1]),
                    self._stats("depths", depths),
                    self._stats("covs2D", covs2D),
                    self._stats("det(covs2D)", det2d),
                    f"valid_depth_count={valid_depth_count}/{N}",
                    f"in_image_center_count={in_image_count}/{N}",
                    f"valid_and_in_image_center_count={visible_center_count}/{N}",
                ],
            )

        # 3. Sort by depth, front-to-back. Invalid depths are still kept but masked out later.
        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        valid_depth_mask = valid_depth_mask[indices]

        # 4. Compute Gaussian values.
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, pixels)  # (N, H, W)
        self._check_finite("gaussian_values", gaussian_values)

        # 5. Apply depth mask.
        gaussian_values = gaussian_values * valid_depth_mask.view(N, 1, 1)

        # 6. Alpha composition setup.
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        self._check_finite("alphas_before_clamp", alphas)
        alphas = alphas.clamp(min=0.0, max=0.99)

        colors_img = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)
        colors_img = colors_img.permute(0, 2, 3, 1)  # (N, H, W, 3)

        if self._should_debug():
            if gaussian_values.max().item() <= self.tiny_threshold:
                self._warn_or_raise(
                    "gaussian_values.max() is tiny. Likely causes: means2D far outside image, "
                    "covs2D too small, invalid covariance, or PDF normalization shrinking values."
                )
            if alphas.max().item() <= self.tiny_threshold:
                self._warn_or_raise(
                    "alphas.max() is tiny. Likely causes: tiny gaussian_values or very small opacities."
                )
            self._print_debug_block(
                "splat values",
                [
                    self._stats("gaussian_values", gaussian_values),
                    self._stats("alphas", alphas),
                    self._stats("colors_sorted", colors),
                    self._stats("opacities_sorted", opacities),
                ],
            )

        # Front-to-back transmittance:
        # T_i = prod_{j < i} (1 - alpha_j)
        ones = torch.ones((1, self.H, self.W), device=alphas.device, dtype=alphas.dtype)
        transmittance = torch.cumprod(torch.cat([ones, 1.0 - alphas + 1e-10], dim=0), dim=0)[:-1]
        weights = alphas * transmittance  # (N, H, W)
        self._check_finite("weights", weights)

        # 8. Final rendering.
        rendered = (weights.unsqueeze(-1) * colors_img).sum(dim=0)  # (H, W, 3)
        rendered = torch.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0)

        if self._should_debug():
            if weights.max().item() <= self.tiny_threshold:
                self._warn_or_raise("weights.max() is tiny, so the final rendered image will be nearly blank.")
            if rendered.max().item() <= self.tiny_threshold:
                self._warn_or_raise(
                    "rendered.max() is tiny. The image is effectively blank; use the debug stats above to locate where values vanished."
                )
            self._print_debug_block(
                "composition output",
                [
                    self._stats("transmittance", transmittance),
                    self._stats("weights", weights),
                    self._stats("rendered", rendered),
                ],
            )

        return rendered
