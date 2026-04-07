import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

# Initialize the polygon state
def initialize_polygon():
    return {'points': [], 'closed': False}

def add_point(img_original, polygon_state, evt: gr.SelectData):
    if polygon_state['closed']:
        return img_original, polygon_state

    x, y = evt.index
    polygon_state['points'].append((x, y))

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)

    for point in polygon_state['points']:
        draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill='blue')

    return img_with_poly, polygon_state

def close_polygon(img_original, polygon_state):
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    else:
        return img_original, polygon_state

def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is None:
        return None

    if polygon_state['closed']:
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + dx, y + dy) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    else:
        return background_image_original

# --------- 改 1：mask 生成 ---------
def create_mask_from_points(points, img_h, img_w):
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    polygon = [tuple(map(int, p)) for p in points.tolist()]
    if len(polygon) >= 3:
        draw.polygon(polygon, outline=255, fill=255)
    return np.array(mask_img, dtype=np.uint8)

# --------- 新增：把前景搬到背景坐标系 ---------
def shift_foreground_to_background(fg_img_tensor, fg_mask_tensor, bg_h, bg_w, dx, dy):
    shifted_fg_img = torch.zeros((1, 3, bg_h, bg_w), device=fg_img_tensor.device, dtype=fg_img_tensor.dtype)
    shifted_fg_mask = torch.zeros((1, 1, bg_h, bg_w), device=fg_img_tensor.device, dtype=fg_img_tensor.dtype)

    ys, xs = torch.where(fg_mask_tensor[0, 0] > 0.5)
    ys2 = ys + int(dy)
    xs2 = xs + int(dx)

    valid = (ys2 >= 0) & (ys2 < bg_h) & (xs2 >= 0) & (xs2 < bg_w)
    ys, xs = ys[valid], xs[valid]
    ys2, xs2 = ys2[valid], xs2[valid]

    shifted_fg_img[0, :, ys2, xs2] = fg_img_tensor[0, :, ys, xs]
    shifted_fg_mask[0, 0, ys2, xs2] = 1.0

    return shifted_fg_img, shifted_fg_mask

# --------- 新增：腐蚀，取内部区域和边界环 ---------
def erode_mask(mask):
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)
    return (F.conv2d(mask, kernel, padding=1) == 9).float()

def get_boundary_ring(mask):
    inner = erode_mask(mask)
    ring = (mask - inner).clamp(0.0, 1.0)
    return ring, inner

# --------- 改 2：loss 加入边界约束 ---------
def cal_laplacian_loss(foreground_img, foreground_mask, blended_img, background_img, background_mask,
                       lambda_boundary=50.0, lambda_color=2.0):
    lap_kernel = torch.tensor(
        [[0., -1., 0.],
         [-1., 4., -1.],
         [0., -1., 0.]],
        device=foreground_img.device,
        dtype=foreground_img.dtype
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    fg_lap = F.conv2d(foreground_img, lap_kernel, padding=1, groups=3)
    blended_lap = F.conv2d(blended_img, lap_kernel, padding=1, groups=3)

    boundary_ring, inner_mask = get_boundary_ring(background_mask)
    inner_mask_3 = inner_mask.expand(-1, 3, -1, -1)
    boundary_ring_3 = boundary_ring.expand(-1, 3, -1, -1)

    # 1. 内部结构保持：Poisson/Laplacian
    poisson_loss = ((blended_lap - fg_lap) ** 2 * inner_mask_3).sum() / (inner_mask_3.sum() + 1e-8)

    # 2. 边界贴住背景，防止白边黑边
    boundary_loss = ((blended_img - background_img) ** 2 * boundary_ring_3).sum() / (boundary_ring_3.sum() + 1e-8)

    # 3. 内部颜色不要漂得太厉害，保持一定颜色一致性
    color_loss = ((blended_img - foreground_img) ** 2 * inner_mask_3).sum() / (inner_mask_3.sum() + 1e-8)

    return poisson_loss + lambda_boundary * boundary_loss + lambda_color * color_loss

# --------- 改 3：blending 主流程 ---------
def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    if not polygon_state['closed'] or background_image_original is None or foreground_image_original is None:
        return background_image_original

    foreground_np = np.array(foreground_image_original.convert("RGB"))
    background_np = np.array(background_image_original.convert("RGB"))

    foreground_polygon_points = np.array(polygon_state['points']).astype(np.int64)
    background_polygon_points = foreground_polygon_points + np.array([int(dx), int(dy)]).reshape(1, 2)

    foreground_mask = create_mask_from_points(foreground_polygon_points, foreground_np.shape[0], foreground_np.shape[1])
    background_mask = create_mask_from_points(background_polygon_points, background_np.shape[0], background_np.shape[1])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fg_img_tensor = torch.from_numpy(foreground_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    bg_img_tensor = torch.from_numpy(background_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    fg_mask_tensor = torch.from_numpy(foreground_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.
    bg_mask_tensor = torch.from_numpy(background_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.

    # 关键：把前景平移到背景坐标系
    shifted_fg_img, shifted_fg_mask = shift_foreground_to_background(
        fg_img_tensor, fg_mask_tensor, background_np.shape[0], background_np.shape[1], dx, dy
    )

    # 初始化：直接从背景开始，不再做0.9/0.1混合
    blended_img = bg_img_tensor.clone()

    # 也可以改成把source patch拷进mask里当初值：
    mask_expanded = bg_mask_tensor.bool().expand(-1, 3, -1, -1)
    blended_img[mask_expanded] = shifted_fg_img[mask_expanded]

    blended_img.requires_grad_(True)
    optimizer = torch.optim.Adam([blended_img], lr=1e-2)

    iter_num = 3000
    for step in range(iter_num):
        blended_img_for_loss = blended_img.detach() * (1. - bg_mask_tensor) + blended_img * bg_mask_tensor

        loss = cal_laplacian_loss(
            shifted_fg_img,
            shifted_fg_mask,
            blended_img_for_loss,
            bg_img_tensor,
            bg_mask_tensor,
            lambda_boundary=100.0,
            lambda_color=2.0
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            blended_img.clamp_(0.0, 1.0)

        if step % 50 == 0:
            print(f'Optimize step: {step}, Loss: {loss.item()}')

        if step == int(iter_num * 2 / 3):
            optimizer.param_groups[0]['lr'] *= 0.1

    result = torch.clamp(blended_img.detach(), 0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    return result.astype(np.uint8)



# Helper function to close the polygon and reset dx
def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    """
    Closes the polygon, resets dx to 0, and updates the background image.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.
        background_image_original (PIL.Image): The original background image.

    Returns:
        tuple: Updated image with polygon, updated polygon state, updated background image, and reset dx value.
    """
    # Close polygon
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)

    # Reset dx value to 0
    new_dx = gr.update(value=0)

    # Update background image
    updated_background = update_background(background_image_original, updated_polygon_state, 0, dy)
    return img_with_poly, updated_polygon_state, updated_background, new_dx

# Gradio Interface
with gr.Blocks(title="Poisson Image Blending", css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
""") as demo:
    # Initialize states
    polygon_state = gr.State(initialize_polygon())
    background_image_original = gr.State(value=None)

    # Title and description
    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Foreground Image")
            foreground_image_original = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>"
            )
            gr.Markdown("### Foreground Image with Polygon")
            foreground_image_with_polygon = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>"
            )
            close_polygon_button = gr.Button("Close Polygon")
        with gr.Column():
            gr.Markdown("### Background Image")
            background_image = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown("<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Background Image with Polygon Overlay")
            background_image_with_polygon = gr.Image(
                label="", type="pil", height=500
            )
            gr.Markdown("<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>")
        with gr.Column():
            gr.Markdown("### Blended Image")
            output_image = gr.Image(
                label="", type="pil", height=500  # Increased height for larger display
            )

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(
                label="Horizontal Offset", minimum=-500, maximum=500, step=1, value=0
            )
        with gr.Column():
            dy = gr.Slider(
                label="Vertical Offset", minimum=-500, maximum=500, step=1, value=0
            )
        blend_button = gr.Button("Blend Images")

    # Interactions

    # Copy the original image to the interactive image when uploaded
    foreground_image_original.change(
        fn=lambda img: img,
        inputs=foreground_image_original,
        outputs=foreground_image_with_polygon,
    )

    # User interacts with the image with polygon
    foreground_image_with_polygon.select(
        add_point,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    close_polygon_button.click(
        fn=close_polygon_and_reset_dx,
        inputs=[foreground_image_original, polygon_state, dx, dy, background_image_original],
        outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx],
    )

    background_image.change(
        fn=lambda img: img,
        inputs=background_image,
        outputs=background_image_original,
    )

    # Update background image when dx or dy changes
    dx.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )
    dy.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    # Blend images when button is clicked
    blend_button.click(
        fn=blending,
        inputs=[foreground_image_original, background_image_original, dx, dy, polygon_state],
        outputs=output_image,
    )

# Launch the Gradio app
demo.launch()