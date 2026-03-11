import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image


def solve_rbf_weights(ctrl_pts, values, eps=1e-8):
    """
    Solve RBF interpolation weights for one displacement component.
    phi(r) = r^2 * log(r + eps)
    ctrl_pts: (N, 2)
    values:   (N,)
    """
    n = ctrl_pts.shape[0]
    diff = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]   # (N, N, 2)
    r = np.linalg.norm(diff, axis=2) + eps               # (N, N)
    K = (r ** 2) * np.log(r + eps)                       # Thin-plate-like RBF
    K += 1e-6 * np.eye(n, dtype=np.float32)              # numerical stability
    w = np.linalg.solve(K, values.astype(np.float32))
    return w.astype(np.float32)


def eval_rbf_field(grid_x, grid_y, ctrl_pts, weights, eps=1e-8):
    """
    Evaluate dense RBF field on all pixels.
    grid_x, grid_y: (H, W)
    ctrl_pts: (N, 2)
    weights: (N,)
    """
    px = ctrl_pts[:, 0].reshape(1, 1, -1)
    py = ctrl_pts[:, 1].reshape(1, 1, -1)

    dx = grid_x[:, :, None] - px
    dy = grid_y[:, :, None] - py
    r = np.sqrt(dx * dx + dy * dy) + eps
    phi = (r ** 2) * np.log(r + eps)

    field = np.sum(phi * weights.reshape(1, 1, -1), axis=2)
    return field.astype(np.float32)


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Use RBF-based deformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    source_pts : (N, 2)
        Original control points.
    target_pts : (N, 2)
        Target control points.
    alpha : float
        Strength of deformation.
    eps : float
        Small value for numerical stability.

    Return
    ------
    warped_image : ndarray
        A deformed image.
    """
    if image is None:
        return None

    warped_image = np.array(image)

    # keep only matched pairs
    n = min(len(source_pts), len(target_pts))
    if n == 0:
        return warped_image

    source_pts = np.array(source_pts[:n], dtype=np.float32)
    target_pts = np.array(target_pts[:n], dtype=np.float32)

    # displacement defined on control points
    disp = (target_pts - source_pts) * alpha
    disp_x = disp[:, 0]
    disp_y = disp[:, 1]

    # solve RBF weights from source control points
    wx = solve_rbf_weights(source_pts, disp_x, eps=eps)
    wy = solve_rbf_weights(source_pts, disp_y, eps=eps)

    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # forward displacement field defined over target image coordinates
    field_x = eval_rbf_field(grid_x, grid_y, source_pts, wx, eps=eps)
    field_y = eval_rbf_field(grid_x, grid_y, source_pts, wy, eps=eps)

    # backward mapping for cv2.remap:
    # output(x, y) samples from input(x - dx, y - dy)
    map_x = (grid_x - field_x).astype(np.float32)
    map_y = (grid_y - field_y).astype(np.float32)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image
    warped_image = point_guided_deformation(
        image,
        np.array(points_src, dtype=np.float32),
        np.array(points_dst, dtype=np.float32)
    )
    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()