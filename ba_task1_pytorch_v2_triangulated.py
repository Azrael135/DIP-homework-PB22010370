"""
Assignment 3 - Task 1: Bundle Adjustment with PyTorch, stabilized version.

Compared with the simple version, this script adds:
  1) triangulation-based 3D point initialization from 2D correspondences;
  2) automatic yaw-direction selection;
  3) staged optimization and mild regularization to avoid a balloon / bulged point cloud;
  4) the same output format used by ba_gradio_viewer.py.

Run example:
  python ba_task1_pytorch_v2_triangulated.py --data_dir data --out_dir outputs_ba_v2 --device cuda --iters 5000
Then visualize:
  python ba_gradio_viewer.py --data_dir data --out_dir outputs_ba_v2 --server_port 7860
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# Data loading
# -------------------------
def load_observations(points2d_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    data = np.load(points2d_path)
    keys = sorted(data.files)
    arrays = []
    for k in keys:
        arr = data[k].astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{k} should have shape (N, 3), but got {arr.shape}")
        arrays.append(arr)
    obs = np.stack(arrays, axis=0)
    xy = torch.from_numpy(obs[..., :2])
    vis = torch.from_numpy(obs[..., 2] > 0.5)
    return xy, vis, keys


def load_colors(color_path: str, num_points: int) -> np.ndarray:
    if os.path.exists(color_path):
        colors = np.load(color_path).astype(np.float32)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"points3d_colors.npy should have shape (N, 3), got {colors.shape}")
        if len(colors) != num_points:
            out = np.ones((num_points, 3), dtype=np.float32) * 255.0
            m = min(num_points, len(colors))
            out[:m] = colors[:m]
            colors = out
    else:
        colors = np.ones((num_points, 3), dtype=np.float32) * 255.0
    if colors.max() > 1.5:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


# -------------------------
# Geometry: NumPy and PyTorch versions must match
# -------------------------
def euler_xyz_to_matrix_np(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float32)
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    shape = euler.shape[:-1]
    Rx = np.zeros(shape + (3, 3), dtype=np.float32)
    Ry = np.zeros(shape + (3, 3), dtype=np.float32)
    Rz = np.zeros(shape + (3, 3), dtype=np.float32)

    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cx
    Rx[..., 1, 2] = -sx
    Rx[..., 2, 1] = sx
    Rx[..., 2, 2] = cx

    Ry[..., 0, 0] = cy
    Ry[..., 0, 2] = sy
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sy
    Ry[..., 2, 2] = cy

    Rz[..., 0, 0] = cz
    Rz[..., 0, 1] = -sz
    Rz[..., 1, 0] = sz
    Rz[..., 1, 1] = cz
    Rz[..., 2, 2] = 1.0
    return Rz @ Ry @ Rx


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sx], dim=-1),
        torch.stack([zeros, sx, cx], dim=-1),
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1),
    ], dim=-2)
    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=-1),
        torch.stack([sz, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)
    return Rz @ Ry @ Rx


def project_points_np(points3d: np.ndarray, euler: np.ndarray, trans: np.ndarray, focal: float, image_size: int) -> np.ndarray:
    R = euler_xyz_to_matrix_np(euler)
    Xc = np.einsum("vij,nj->vni", R, points3d) + trans[:, None, :]
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    z_safe = np.where(np.abs(z) < 1e-6, -1e-6, z)
    cx = cy = image_size * 0.5
    u = -focal * x / z_safe + cx
    v = focal * y / z_safe + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def project_points(points3d: torch.Tensor, euler: torch.Tensor, trans: torch.Tensor, focal: torch.Tensor,
                   image_size: int = 1024, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    R = euler_xyz_to_matrix(euler)
    Xc = torch.einsum("vij,nj->vni", R, points3d) + trans[:, None, :]
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    # In this assignment, valid object points should have negative camera-space z.
    z_safe = torch.where(z.abs() < eps, torch.full_like(z, -eps), z)
    cx = cy = image_size * 0.5
    u = -focal * x / z_safe + cx
    v = focal * y / z_safe + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, z


def smooth_l1_reprojection(pred_uv: torch.Tensor, obs_uv: torch.Tensor, vis: torch.Tensor, beta: float) -> torch.Tensor:
    residual = pred_uv - obs_uv
    residual = residual[vis]
    if residual.numel() == 0:
        raise RuntimeError("No visible observations found.")
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=beta, reduction="mean")


# -------------------------
# Initialization
# -------------------------
def initial_cameras(num_views: int, init_yaw_deg: float, yaw_sign: float, init_depth: float) -> Tuple[np.ndarray, np.ndarray]:
    euler = np.zeros((num_views, 3), dtype=np.float32)
    if init_yaw_deg > 0:
        yaw = np.linspace(-math.radians(init_yaw_deg), math.radians(init_yaw_deg), num_views).astype(np.float32)
        euler[:, 1] = yaw_sign * yaw
    trans = np.zeros((num_views, 3), dtype=np.float32)
    trans[:, 2] = -float(init_depth)
    return euler, trans


def triangulate_points_linear(obs_uv: np.ndarray, vis: np.ndarray, euler: np.ndarray, trans: np.ndarray,
                              focal: float, image_size: int, max_views_per_point: int,
                              seed: int) -> np.ndarray:
    """Linear triangulation under the assignment projection convention.

    For each visible observation:
      (u-cx) * Zc + f * Xc = 0
      (v-cy) * Zc - f * Yc = 0
    where Xc = r1 P + tx, Yc = r2 P + ty, Zc = r3 P + tz.
    """
    rng = np.random.default_rng(seed)
    V, N = obs_uv.shape[:2]
    R = euler_xyz_to_matrix_np(euler)
    cx = cy = image_size * 0.5
    points = np.zeros((N, 3), dtype=np.float32)

    for n in range(N):
        views = np.where(vis[:, n])[0]
        if len(views) < 2:
            points[n] = 0.2 * rng.standard_normal(3)
            points[n, 2] *= 0.3
            continue

        if len(views) > max_views_per_point:
            # Use evenly distributed views rather than the first few views.
            take = np.linspace(0, len(views) - 1, max_views_per_point).astype(np.int64)
            views = views[take]

        rows = []
        rhs = []
        for v in views:
            u, vv = obs_uv[v, n]
            du = float(u - cx)
            dv = float(vv - cy)
            r1, r2, r3 = R[v, 0], R[v, 1], R[v, 2]
            tx, ty, tz = trans[v]

            rows.append(du * r3 + focal * r1)
            rhs.append(-(du * tz + focal * tx))
            rows.append(dv * r3 - focal * r2)
            rhs.append(-(dv * tz - focal * ty))

        A = np.asarray(rows, dtype=np.float32)
        b = np.asarray(rhs, dtype=np.float32)
        try:
            p, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            p = 0.2 * rng.standard_normal(3)
            p[2] *= 0.3

        if not np.all(np.isfinite(p)) or np.linalg.norm(p) > 10.0:
            p = 0.2 * rng.standard_normal(3)
            p[2] *= 0.3
        points[n] = p.astype(np.float32)

    # Remove a small global translation drift, but keep scale.
    med = np.median(points, axis=0, keepdims=True)
    points = points - med
    return points.astype(np.float32)


def initial_mean_l2(obs_uv: np.ndarray, vis: np.ndarray, pred_uv: np.ndarray, sample_points: int = 4000) -> float:
    V, N = vis.shape
    if sample_points < N:
        idx = np.linspace(0, N - 1, sample_points).astype(np.int64)
    else:
        idx = np.arange(N)
    mask = vis[:, idx]
    if mask.sum() == 0:
        return float("inf")
    residual = pred_uv[:, idx, :] - obs_uv[:, idx, :]
    l2 = np.linalg.norm(residual[mask], axis=1)
    return float(np.mean(l2))


def build_initial_parameters(obs_uv_cpu: torch.Tensor, vis_cpu: torch.Tensor, args):
    obs_np = obs_uv_cpu.numpy().astype(np.float32)
    vis_np = vis_cpu.numpy().astype(bool)
    num_views, num_points = obs_np.shape[:2]

    if args.init_method == "random":
        rng = np.random.default_rng(args.seed)
        pts = 0.25 * rng.standard_normal((num_points, 3)).astype(np.float32)
        pts[:, 2] *= 0.5
        euler, trans = initial_cameras(num_views, args.init_yaw_deg, args.yaw_sign, args.init_depth)
        return pts, euler, trans

    candidates = [args.yaw_sign]
    if args.auto_yaw_sign:
        candidates = [1.0, -1.0]

    best = None
    for yaw_sign in candidates:
        print(f"[Init] triangulating points with yaw_sign={yaw_sign:+.0f} ...")
        euler, trans = initial_cameras(num_views, args.init_yaw_deg, yaw_sign, args.init_depth)
        pts = triangulate_points_linear(
            obs_np, vis_np, euler, trans,
            focal=args.init_focal,
            image_size=args.image_size,
            max_views_per_point=args.max_triangulation_views,
            seed=args.seed,
        )
        pred = project_points_np(pts, euler, trans, args.init_focal, args.image_size)
        err = initial_mean_l2(obs_np, vis_np, pred)
        print(f"[Init] yaw_sign={yaw_sign:+.0f}, initial mean L2 reprojection error = {err:.3f} px")
        if best is None or err < best[0]:
            best = (err, pts, euler, trans, yaw_sign)

    assert best is not None
    print(f"[Init] selected yaw_sign={best[4]:+.0f}")
    return best[1], best[2], best[3]


# -------------------------
# Output
# -------------------------
def save_obj(path: str, points: np.ndarray, colors: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Reconstructed point cloud from PyTorch Bundle Adjustment\n")
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def plot_loss(losses, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection loss / px")
    plt.title("Bundle Adjustment Optimization Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Training
# -------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[Info] using device: {device}")

    data_dir = Path(args.data_dir)
    obs_uv_cpu, vis_cpu, keys = load_observations(str(data_dir / "points2d.npz"))
    num_views, num_points = obs_uv_cpu.shape[:2]
    colors = load_colors(str(data_dir / "points3d_colors.npy"), num_points)

    print(f"[Info] loaded observations: views={num_views}, points={num_points}")
    print(f"[Info] visible observations: {int(vis_cpu.sum())} / {vis_cpu.numel()}")

    init_points, init_euler, init_trans = build_initial_parameters(obs_uv_cpu, vis_cpu, args)

    obs_uv = obs_uv_cpu.to(device=device, dtype=torch.float32)
    vis = vis_cpu.to(device=device)

    points = torch.nn.Parameter(torch.from_numpy(init_points).to(device=device, dtype=torch.float32))
    euler = torch.nn.Parameter(torch.from_numpy(init_euler).to(device=device, dtype=torch.float32))
    trans = torch.nn.Parameter(torch.from_numpy(init_trans).to(device=device, dtype=torch.float32))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(args.init_focal), dtype=torch.float32, device=device))

    euler_prior = torch.from_numpy(init_euler).to(device=device, dtype=torch.float32)
    trans_prior = torch.from_numpy(init_trans).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([
        {"params": [points], "lr": args.lr_points},
        {"params": [euler], "lr": args.lr_rot},
        {"params": [trans], "lr": args.lr_trans},
        {"params": [log_focal], "lr": args.lr_focal},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.iters // 3), gamma=0.5)

    losses = []
    for it in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)

        # Stage 1: refine cameras first while points are almost fixed.
        if it <= args.stage1_iters:
            points.requires_grad_(False)
        else:
            points.requires_grad_(True)

        focal = torch.exp(log_focal).clamp(args.min_focal, args.max_focal)
        pred_uv, zc = project_points(points, euler, trans, focal, image_size=args.image_size)
        reproj = smooth_l1_reprojection(pred_uv, obs_uv, vis, beta=args.huber_delta)

        # Mild penalties to avoid degenerate/bulged solutions.
        depth_penalty = F.relu(zc[vis] + args.depth_margin).mean()
        center_penalty = points.mean(dim=0).pow(2).sum()
        radius = points.norm(dim=1)
        radius_penalty = F.relu(radius - args.max_point_radius).pow(2).mean()
        cam_prior = (euler - euler_prior).pow(2).mean() + 0.1 * (trans - trans_prior).pow(2).mean()
        trans_xy_penalty = trans[:, :2].pow(2).mean()

        loss = (
            reproj
            + args.lambda_depth * depth_penalty
            + args.lambda_center * center_penalty
            + args.lambda_radius * radius_penalty
            + args.lambda_cam_prior * cam_prior
            + args.lambda_trans_xy * trans_xy_penalty
        )
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in [points, euler, trans, log_focal] if p.grad is not None], args.grad_clip)
        optimizer.step()
        scheduler.step()

        losses.append(float(reproj.detach().cpu()))
        if it == 1 or it % args.log_every == 0 or it == args.iters:
            with torch.no_grad():
                res = pred_uv[vis] - obs_uv[vis]
                mean_l2 = torch.linalg.norm(res, dim=1).mean().item()
                median_l2 = torch.linalg.norm(res, dim=1).median().item()
                f_val = float(torch.exp(log_focal).detach().cpu())
                print(
                    f"iter {it:05d}/{args.iters} | "
                    f"huber={reproj.item():.4f} | mean_l2={mean_l2:.3f}px | "
                    f"median_l2={median_l2:.3f}px | f={f_val:.2f} | "
                    f"stage={'camera' if it <= args.stage1_iters else 'full'}"
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_np = points.detach().cpu().numpy()
    euler_np = euler.detach().cpu().numpy()
    trans_np = trans.detach().cpu().numpy()
    focal_np = np.array([float(torch.exp(log_focal).detach().cpu())], dtype=np.float32)

    save_obj(str(out_dir / "reconstruction.obj"), points_np, colors)
    plot_loss(losses, str(out_dir / "loss_curve.png"))
    np.savez(
        out_dir / "optimized_params.npz",
        points3d=points_np,
        euler=euler_np,
        trans=trans_np,
        focal=focal_np,
        view_keys=np.array(keys),
        loss=np.array(losses, dtype=np.float32),
    )

    print(f"[Done] saved OBJ:        {out_dir / 'reconstruction.obj'}")
    print(f"[Done] saved loss curve: {out_dir / 'loss_curve.png'}")
    print(f"[Done] saved params:     {out_dir / 'optimized_params.npz'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stabilized PyTorch Bundle Adjustment for Assignment 3 Task 1")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs_ba_v2")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--stage1_iters", type=int, default=800)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--init_method", type=str, default="triangulate", choices=["triangulate", "random"])
    parser.add_argument("--init_depth", type=float, default=2.5)
    parser.add_argument("--init_focal", type=float, default=900.0)
    parser.add_argument("--min_focal", type=float, default=200.0)
    parser.add_argument("--max_focal", type=float, default=3000.0)
    parser.add_argument("--init_yaw_deg", type=float, default=70.0)
    parser.add_argument("--yaw_sign", type=float, default=1.0, choices=[1.0, -1.0])
    parser.add_argument("--auto_yaw_sign", action="store_true", default=True)
    parser.add_argument("--max_triangulation_views", type=int, default=15)

    parser.add_argument("--lr_points", type=float, default=5e-3)
    parser.add_argument("--lr_rot", type=float, default=8e-4)
    parser.add_argument("--lr_trans", type=float, default=1e-3)
    parser.add_argument("--lr_focal", type=float, default=1e-4)
    parser.add_argument("--huber_delta", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--depth_margin", type=float, default=0.05)
    parser.add_argument("--max_point_radius", type=float, default=1.5)
    parser.add_argument("--lambda_depth", type=float, default=1e-2)
    parser.add_argument("--lambda_center", type=float, default=1e-3)
    parser.add_argument("--lambda_radius", type=float, default=1e-3)
    parser.add_argument("--lambda_cam_prior", type=float, default=1e-4)
    parser.add_argument("--lambda_trans_xy", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
