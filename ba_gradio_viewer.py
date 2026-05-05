"""
Interactive Gradio viewer for Assignment 3 - Task 1 Bundle Adjustment.

Use after running ba_task1_pytorch_kmpfix.py, which should produce:
    outputs_ba/optimized_params.npz
    outputs_ba/loss_curve.png
    outputs_ba/reconstruction.obj

Run:
    python ba_gradio_viewer.py --data_dir data --out_dir outputs_ba --server_port 7860

What it shows:
    1) Interactive 3D reconstructed point cloud
    2) Optimization loss curve
    3) Per-view observed 2D points vs reprojected 2D points overlay

Dependencies:
    pip install gradio plotly pillow numpy
"""

import os
# Safe workaround for common Windows/Conda OpenMP conflicts. Must appear before numpy/matplotlib-like imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw


# -------------------------
# Loading utilities
# -------------------------
def sorted_image_files(image_dir: Path):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(glob.glob(str(image_dir / ext)))
    return sorted(files)


def load_points2d(points2d_file: Path):
    if not points2d_file.exists():
        raise FileNotFoundError(f"Cannot find {points2d_file}. Please run from the assignment root or pass --data_dir.")
    data = np.load(points2d_file)
    keys = sorted(data.files)
    points_dict = {}
    for k in keys:
        arr = data[k].astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{k} should have shape (N, 3), got {arr.shape}")
        points_dict[k] = arr
    return points_dict, keys


def load_colors(color_file: Path, num_points: int):
    if color_file.exists():
        colors = np.load(color_file).astype(np.float32)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"points3d_colors.npy should have shape (N, 3), got {colors.shape}")
        if len(colors) != num_points:
            out = np.ones((num_points, 3), dtype=np.float32) * 255.0
            m = min(num_points, len(colors))
            out[:m] = colors[:m]
            colors = out
    else:
        colors = np.ones((num_points, 3), dtype=np.float32) * 255.0

    if colors.max() <= 1.5:
        colors_255 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        colors_01 = np.clip(colors, 0, 1)
    else:
        colors_255 = np.clip(colors, 0, 255).astype(np.uint8)
        colors_01 = colors_255.astype(np.float32) / 255.0
    return colors_01, colors_255


def load_params(out_dir: Path):
    params_file = out_dir / "optimized_params.npz"
    if not params_file.exists():
        raise FileNotFoundError(
            f"Cannot find {params_file}. Run BA first, for example:\n"
            f"python ba_task1_pytorch_kmpfix.py --data_dir data --out_dir {out_dir} --device cuda --iters 4000"
        )
    params = np.load(params_file, allow_pickle=True)
    required = ["points3d", "euler", "trans", "focal"]
    for k in required:
        if k not in params.files:
            raise KeyError(f"{params_file} is missing key: {k}")
    return params


# -------------------------
# Geometry, same convention as the PyTorch BA script
# -------------------------
def euler_xyz_to_matrix_np(euler: np.ndarray):
    """R = Rz @ Ry @ Rx, matching the BA script."""
    euler = np.asarray(euler, dtype=np.float32)
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    shape = euler.shape[:-1]
    Rx = np.zeros(shape + (3, 3), dtype=np.float32)
    Ry = np.zeros(shape + (3, 3), dtype=np.float32)
    Rz = np.zeros(shape + (3, 3), dtype=np.float32)

    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = cx
    Rx[..., 1, 2] = -sx
    Rx[..., 2, 1] = sx
    Rx[..., 2, 2] = cx

    Ry[..., 0, 0] = cy
    Ry[..., 0, 2] = sy
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = -sy
    Ry[..., 2, 2] = cy

    Rz[..., 0, 0] = cz
    Rz[..., 0, 1] = -sz
    Rz[..., 1, 0] = sz
    Rz[..., 1, 1] = cz
    Rz[..., 2, 2] = 1

    return Rz @ Ry @ Rx


def project_all_views(points3d: np.ndarray, euler: np.ndarray, trans: np.ndarray, focal: float, image_size: int):
    R = euler_xyz_to_matrix_np(euler)  # (V, 3, 3)
    Xc = np.einsum("vij,nj->vni", R, points3d) + trans[:, None, :]
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    z_safe = np.where(np.abs(z) < 1e-6, -1e-6, z)
    cx = cy = image_size * 0.5
    u = -focal * x / z_safe + cx
    v = focal * y / z_safe + cy
    uv = np.stack([u, v], axis=-1)
    return uv, Xc[..., 2]


# -------------------------
# Visualization helpers
# -------------------------
def rgb_strings(colors_255: np.ndarray):
    c = np.clip(colors_255, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def make_pointcloud_fig(points3d, colors_255, max_points=8000, point_size=2):
    n = len(points3d)
    max_points = int(max(100, min(max_points, n)))
    if max_points < n:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    else:
        idx = np.arange(n)

    pts = points3d[idx]
    cols = rgb_strings(colors_255[idx])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=cols, opacity=0.9),
                text=[f"point {i}" for i in idx],
                hoverinfo="text+x+y+z",
            )
        ]
    )
    fig.update_layout(
        title=f"Reconstructed 3D Point Cloud ({len(idx)} / {n} points shown)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def make_loss_fig(loss):
    # Kept for compatibility, but the Gradio tab below uses a PNG/PIL image instead.
    # Some Windows/Gradio/Plotly combinations fail to render a simple 2D Plotly figure
    # even when the 3D Plotly figure works, so image display is more robust.
    if loss is None or len(loss) == 0:
        fig = go.Figure()
        fig.update_layout(title="No loss saved in optimized_params.npz")
        return fig
    x = np.arange(1, len(loss) + 1)
    fig = go.Figure(data=[go.Scatter(x=x, y=loss, mode="lines", name="reprojection loss")])
    fig.update_layout(
        title="Bundle Adjustment Optimization Loss",
        xaxis_title="Iteration",
        yaxis_title="Reprojection loss / px",
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_loss_image(out_dir: Path, loss):
    """Return a PIL image for the loss curve. Prefer the PNG saved by the optimizer.

    This avoids a common Gradio/Plotly rendering error in the loss tab on Windows.
    """
    png_path = out_dir / "loss_curve.png"
    if png_path.exists():
        return Image.open(png_path).convert("RGB")

    # Fallback: draw a simple loss curve using only PIL, no matplotlib dependency.
    W, H = 1000, 420
    margin_l, margin_r, margin_t, margin_b = 80, 30, 40, 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 12), "Bundle Adjustment Optimization Loss", fill=(0, 0, 0))

    if loss is None or len(loss) == 0:
        draw.text((margin_l, 180), "No loss_curve.png and no loss array found in optimized_params.npz", fill=(200, 0, 0))
        return img

    y = np.asarray(loss, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        draw.text((margin_l, 180), "Loss values are missing or non-finite.", fill=(200, 0, 0))
        return img
    y = y[valid]
    x = np.arange(len(y), dtype=np.float64)

    # Use percentiles to make the plot readable even if the first few losses are huge.
    ymin = float(np.percentile(y, 1))
    ymax = float(np.percentile(y, 99))
    if abs(ymax - ymin) < 1e-12:
        ymin -= 1.0
        ymax += 1.0
    y_clip = np.clip(y, ymin, ymax)

    x0, y0 = margin_l, H - margin_b
    x1, y1 = W - margin_r, margin_t
    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=2)
    draw.line((x0, y0, x0, y1), fill=(0, 0, 0), width=2)
    draw.text((W // 2 - 40, H - 35), "Iteration", fill=(0, 0, 0))
    draw.text((8, H // 2), "Loss", fill=(0, 0, 0))
    draw.text((x0 - 70, y1 - 5), f"{ymax:.3g}", fill=(0, 0, 0))
    draw.text((x0 - 70, y0 - 10), f"{ymin:.3g}", fill=(0, 0, 0))
    draw.text((x0, y0 + 8), "1", fill=(0, 0, 0))
    draw.text((x1 - 70, y0 + 8), str(len(y)), fill=(0, 0, 0))

    xs = x0 + (x / max(len(y) - 1, 1)) * (x1 - x0)
    ys = y0 - ((y_clip - ymin) / (ymax - ymin)) * (y0 - y1)
    pts = list(zip(xs.astype(float), ys.astype(float)))
    if len(pts) > 1:
        draw.line(pts, fill=(30, 90, 200), width=2)
    draw.text((margin_l, H - 55), f"first={y[0]:.6g}, last={y[-1]:.6g}, min={np.min(y):.6g}", fill=(0, 0, 0))
    return img


def draw_point(draw: ImageDraw.ImageDraw, x, y, r, color):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def make_overlay_image(
    image_path,
    obs_arr,
    pred_uv,
    colors_255,
    max_points=1500,
    point_size=3,
    show_lines=True,
    mode="Observed red + Reprojected cyan",
):
    if image_path is not None and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (1024, 1024), (255, 255, 255))

    draw = ImageDraw.Draw(img)
    obs_xy = obs_arr[:, :2]
    vis = obs_arr[:, 2] > 0.5
    visible_idx = np.where(vis)[0]
    if len(visible_idx) == 0:
        return img

    max_points = int(max(50, min(max_points, len(visible_idx))))
    if max_points < len(visible_idx):
        local = np.linspace(0, len(visible_idx) - 1, max_points).astype(np.int64)
        idx = visible_idx[local]
    else:
        idx = visible_idx

    obs = obs_xy[idx]
    pred = pred_uv[idx]

    for pid, (o, p) in zip(idx, zip(obs, pred)):
        ox, oy = float(o[0]), float(o[1])
        px, py = float(p[0]), float(p[1])

        if show_lines:
            draw.line((ox, oy, px, py), fill=(255, 255, 0), width=1)

        if mode == "Use point RGB colors":
            c = tuple(colors_255[pid].tolist())
            draw_point(draw, ox, oy, point_size, c)
            draw_point(draw, px, py, point_size, (0, 255, 255))
        else:
            draw_point(draw, ox, oy, point_size, (255, 0, 0))      # observed
            draw_point(draw, px, py, point_size, (0, 255, 255))    # reprojected

    return img


def compute_view_info(view_key, obs_arr, pred_uv, focal, euler_v, trans_v):
    vis = obs_arr[:, 2] > 0.5
    visible = int(vis.sum())
    total = len(obs_arr)
    if visible > 0:
        residual = pred_uv[vis] - obs_arr[vis, :2]
        l2 = np.linalg.norm(residual, axis=1)
        mean_l2 = float(l2.mean())
        median_l2 = float(np.median(l2))
        p90_l2 = float(np.percentile(l2, 90))
    else:
        mean_l2 = median_l2 = p90_l2 = float("nan")

    euler_deg = np.rad2deg(euler_v)
    info = (
        f"View: {view_key}\n"
        f"Visible points: {visible} / {total}\n"
        f"Shared focal length f: {float(focal):.3f}\n"
        f"Mean reprojection L2 error: {mean_l2:.3f} px\n"
        f"Median reprojection L2 error: {median_l2:.3f} px\n"
        f"90% reprojection L2 error: {p90_l2:.3f} px\n"
        f"Euler angles XYZ: [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}] deg\n"
        f"Translation T: [{trans_v[0]:.4f}, {trans_v[1]:.4f}, {trans_v[2]:.4f}]"
    )
    return info


def build_app(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    points2d_dict, view_keys = load_points2d(data_dir / "points2d.npz")
    images = sorted_image_files(data_dir / "images")

    params = load_params(out_dir)
    points3d = params["points3d"].astype(np.float32)
    euler = params["euler"].astype(np.float32)
    trans = params["trans"].astype(np.float32)
    focal = float(np.asarray(params["focal"]).reshape(-1)[0])
    loss = params["loss"].astype(np.float32) if "loss" in params.files else None

    num_views = len(view_keys)
    num_points = len(points3d)
    colors_01, colors_255 = load_colors(data_dir / "points3d_colors.npy", num_points)

    pred_all, _ = project_all_views(points3d, euler, trans, focal, args.image_size)

    def get_image_path(view_key):
        i = view_keys.index(view_key)
        if i < len(images):
            return images[i]
        return None

    def update_overlay(view_key, max_points, point_size, show_lines, color_mode):
        vi = view_keys.index(view_key)
        obs_arr = points2d_dict[view_key]
        pred_uv = pred_all[vi]
        img = make_overlay_image(
            get_image_path(view_key),
            obs_arr,
            pred_uv,
            colors_255,
            max_points=max_points,
            point_size=point_size,
            show_lines=show_lines,
            mode=color_mode,
        )
        info = compute_view_info(view_key, obs_arr, pred_uv, focal, euler[vi], trans[vi])
        return img, info

    def update_cloud(max_points, point_size):
        return make_pointcloud_fig(points3d, colors_255, max_points=max_points, point_size=point_size)

    with gr.Blocks(title="Bundle Adjustment Result Viewer") as demo:
        gr.Markdown(
            "# Bundle Adjustment Interactive Result Viewer\n"
            "这个页面用于展示 Task 1 的优化结果：loss 曲线、重建 3D 点云、以及每个视角下观测点与重投影点的对比。\n\n"
            "图像叠加图中：**红色 = 真实 2D 观测点**，**青色 = 优化后 3D 点重新投影的位置**，**黄色线段 = 重投影误差**。"
        )

        with gr.Tab("3D reconstruction"):
            with gr.Row():
                max_3d = gr.Slider(500, min(num_points, 20000), value=min(num_points, 8000), step=500, label="3D points shown")
                size_3d = gr.Slider(1, 6, value=2, step=1, label="3D point size")
            cloud_plot = gr.Plot(label="Interactive 3D point cloud")
            refresh_cloud = gr.Button("Update 3D point cloud")
            refresh_cloud.click(update_cloud, inputs=[max_3d, size_3d], outputs=cloud_plot)
            demo.load(update_cloud, inputs=[max_3d, size_3d], outputs=cloud_plot)

        with gr.Tab("Loss curve"):
            #gr.Markdown("如果这里之前显示红色‘错误’，通常是 Gradio/Plotly 的 2D 图渲染问题，不影响优化结果。这里改为直接显示优化脚本保存的 loss_curve.png。")
            loss_img = gr.Image(label="Optimization loss curve", type="pil")
            demo.load(lambda: make_loss_image(out_dir, loss), inputs=None, outputs=loss_img)

        with gr.Tab("2D reprojection check"):
            with gr.Row():
                view_dropdown = gr.Dropdown(choices=view_keys, value=view_keys[0], label="View")
                max_2d = gr.Slider(100, 5000, value=1500, step=100, label="2D points shown")
                size_2d = gr.Slider(1, 8, value=3, step=1, label="2D point size")
            with gr.Row():
                line_checkbox = gr.Checkbox(value=True, label="Show residual lines")
                color_mode = gr.Dropdown(
                    choices=["Observed red + Reprojected cyan", "Use point RGB colors"],
                    value="Observed red + Reprojected cyan",
                    label="Overlay color mode",
                )
            overlay = gr.Image(label="Observed vs reprojected points", type="pil")
            info = gr.Textbox(label="View diagnostics", lines=8)
            refresh_overlay = gr.Button("Update selected view")
            refresh_overlay.click(
                update_overlay,
                inputs=[view_dropdown, max_2d, size_2d, line_checkbox, color_mode],
                outputs=[overlay, info],
            )
            demo.load(
                update_overlay,
                inputs=[view_dropdown, max_2d, size_2d, line_checkbox, color_mode],
                outputs=[overlay, info],
            )

        gr.Markdown(
            f"Loaded `{out_dir / 'optimized_params.npz'}`. "
            f"Views: **{num_views}**, points: **{num_points}**, focal: **{focal:.3f}**."
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio viewer for PyTorch Bundle Adjustment result")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs_ba")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args)
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
