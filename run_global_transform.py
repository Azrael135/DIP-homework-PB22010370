import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]]).astype(np.float32)

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)

    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8
    ) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)

    image_new[pad_size:pad_size + image.shape[0], pad_size:pad_size + image.shape[1]] = image
    image = np.array(image_new)

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # ========== FILL: Apply Composition Transform ==========
    # We compose transforms in homogeneous coordinates:
    # 1) optional horizontal flip around image center
    # 2) scale around image center
    # 3) rotation around image center
    # 4) translation

    # Translation matrices for center-based operations
    T_to_center = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)

    T_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Scale around center
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    M_scale = T_back @ S @ T_to_center

    # Rotation around center
    theta = np.deg2rad(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=np.float32)
    M_rot = T_back @ R @ T_to_center

    # Optional horizontal flip around center
    if flip_horizontal:
        F = np.array([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=np.float32)
        M_flip = T_back @ F @ T_to_center
    else:
        M_flip = np.eye(3, dtype=np.float32)

    # Translation
    M_trans = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Final composition
    # Order: flip -> scale -> rotate -> translate
    M = M_trans @ M_rot @ M_scale @ M_flip

    transformed_image = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()