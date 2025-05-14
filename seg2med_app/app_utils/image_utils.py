# utils/image_utils.py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib

import base64
from io import BytesIO

default_orientation_type = 'transpose'
default_plt_origin_type = 'upper'

     
def image_to_base64(img, width=200):
    buffered = BytesIO()
    if img.mode == "RGBA":
        img.save(buffered, format="PNG")
        result = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="data:image/png;base64,{result}" width="{width}">
            </div>
            """,
            unsafe_allow_html=True
            )
    else:
        img.save(buffered, format="JPEG")
        result = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="data:image/jpeg;base64,{result}" width="{width}">
            </div>
            """,
            unsafe_allow_html=True
            )
        

def processing_slice_to_right_orientation(img_slice, type=default_orientation_type):
    if type == 'transpose':
        return img_slice.T
    elif type == 'rot90':
        return np.rot90(img_slice)
    elif type == 'none':
        return img_slice

def restore_slice_to_wrong_orientation(img_slice, type=default_orientation_type):
    if type == 'transpose':
        return img_slice.T
    elif type == 'rot90':
        return np.rot90(img_slice,3)
    elif type == 'none':
        return img_slice

def load_image_canonical(nii_file):
    img = nib.load(nii_file)
    #img_canonical = nib.as_closest_canonical(img)
    data = img.get_fdata()
    return img
def get_compatible_cmap(name="tab20", N=20):
    # 优先使用 plt.get_cmap()，如果没有 fallback 到 cm.get_cmap()
    try:
        return plt.get_cmap(name, N)
    except TypeError:
        # for older versions of matplotlib
        return cm.get_cmap(name, N)


def generate_color_map(label_ids, cmap='tab20'):
    cmap = get_compatible_cmap(cmap, len(label_ids))  # or 'Set3', 'tab10'
    color_map = {}
    for i, label_id in enumerate(label_ids):
        rgba = cmap(i)
        rgb = tuple(int(255 * c) for c in rgba[:3])
        color_map[label_id] = ",".join(map(str, rgb))
    return color_map

# utils/image_utils.py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def global_slice_slider(volume_shape):
    st.markdown("### 🔎 Global slice controller")
    col_z, col_y, col_x = st.columns(3)
    with col_z:
        z_idx = st.slider("Axial (Z)", 0, volume_shape[2]-1, volume_shape[2] // 2, key="z_slider")
    with col_y:
        y_idx = st.slider("Coronal (Y)", 0, volume_shape[1]-1, volume_shape[1] // 2, key="y_slider")
    with col_x:
        x_idx = st.slider("Sagittal (X)", 0, volume_shape[0]-1, volume_shape[0] // 2, key="x_slider")
    return z_idx, y_idx, x_idx


from PIL import Image
def show_single_slice_label(label2d, label_colors, title="Label Slice"):
    """
    显示单张 2D 标签图像（使用 RGB 映射）。
    label_colors: dict[int -> str]，如 {1: "255,0,0"}
    """
    import matplotlib.pyplot as plt
    import io

    rgb_map = np.zeros((*label2d.shape, 3), dtype=np.uint8)
    for label, rgb_str in label_colors.items():
        rgb_vals = [int(v) for v in rgb_str.split(",")]
        mask = label2d == label
        mask = processing_slice_to_right_orientation(mask)
        for c in range(3):
            rgb_map[:, :, c][mask] = rgb_vals[c]
            
    st.image(Image.fromarray(rgb_map), use_container_width =True)

def show_single_slice_image(image2d, title="Slice", orientation_type=default_orientation_type):
    """
    用 Streamlit 原生方式显示灰度图（不经过 matplotlib）。
    """
    import numpy as np

    # normalize to [0, 255]
    img = image2d.astype(np.float32)
    img = np.nan_to_num(img)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = processing_slice_to_right_orientation(img, orientation_type)
    img_uint8 = (img * 255).astype(np.uint8)

    '''fig, ax = plt.subplots()
    ax.imshow(rgb_map, cmap='gray', origin=default_plt_origin_type)
    ax.axis('off')
    st.pyplot(fig)'''
        
    st.image(img_uint8, caption=title, use_container_width =True, clamp=True)
    
def show_three_planes_interactive(image, z_idx, y_idx, x_idx, orientation_type=default_orientation_type):
    """
    Show three orthogonal planes simultaneously with slice sliders.
    Supports nibabel image object or raw NumPy array.
    """
    if hasattr(image, "get_fdata"):
        data = image.get_fdata()
    else:
        data = image

    data = np.nan_to_num(data).astype(np.float32)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(processing_slice_to_right_orientation(data[:, :, z_idx],orientation_type), cmap='gray', origin=default_plt_origin_type)
    axs[0].set_title(f"Axial @ {z_idx}")

    axs[1].imshow(processing_slice_to_right_orientation(data[:, y_idx, :],orientation_type), cmap='gray', origin=default_plt_origin_type)
    axs[1].set_title(f"Coronal @ {y_idx}")

    axs[2].imshow(processing_slice_to_right_orientation(data[x_idx, :, :],orientation_type), cmap='gray', origin=default_plt_origin_type)
    axs[2].set_title(f"Sagittal @ {x_idx}")

    for ax in axs:
        ax.axis('off')

    st.pyplot(fig)


def show_label_overlay(label_volume, z_idx, y_idx, x_idx, label_colors=None):
    """
    Show label slices in three orthogonal planes with sliders and color overlays.
    label_colors: dict[int -> str] with RGB strings like "255,0,0"
    """
    if hasattr(label_volume, "get_fdata"):
        label_data = label_volume.get_fdata().astype(np.int32)
    else:
        label_data = label_volume.astype(np.int32)

    z_max = label_data.shape[2] - 1
    y_max = label_data.shape[1] - 1
    x_max = label_data.shape[0] - 1

    def label_to_rgb(slice_data):
        if not label_colors:
            return slice_data
        rgb_map = np.zeros((*slice_data.shape, 3), dtype=np.uint8)
        for label, rgb_str in label_colors.items():
            if isinstance(rgb_str, str):
                rgb_vals = [int(v) for v in rgb_str.split(",")]
            else:
                rgb_vals = [0, 0, 0]
            mask = slice_data == label
            for c in range(3):
                rgb_map[:, :, c][mask] = rgb_vals[c]
        return rgb_map

    axial = label_to_rgb(processing_slice_to_right_orientation(label_data[:, :, z_idx]))
    coronal = label_to_rgb(processing_slice_to_right_orientation(label_data[:, y_idx, :]))
    sagittal = label_to_rgb(processing_slice_to_right_orientation(label_data[x_idx, :, :]))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(axial, origin=default_plt_origin_type)
    axs[0].set_title(f"Axial @ {z_idx}")

    axs[1].imshow(coronal, origin=default_plt_origin_type)
    axs[1].set_title(f"Coronal @ {y_idx}")

    axs[2].imshow(sagittal, origin=default_plt_origin_type)
    axs[2].set_title(f"Sagittal @ {x_idx}")

    for ax in axs:
        ax.axis('off')

    st.pyplot(fig)


def show_three_planes(image, title_prefix=""):
    if hasattr(image, "get_fdata"):
        data = image.get_fdata()
    else:
        data = image

    data = np.nan_to_num(data).astype(np.float32)
    mid_axial = data.shape[2] // 2
    mid_coronal = data.shape[1] // 2
    mid_sagittal = data.shape[0] // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(processing_slice_to_right_orientation(data[:, :, mid_axial]), cmap='gray', origin=default_plt_origin_type)
    axs[0].set_title(f'{title_prefix} Axial')

    axs[1].imshow(processing_slice_to_right_orientation(data[:, mid_coronal, :]), cmap='gray', origin=default_plt_origin_type)
    axs[1].set_title(f'{title_prefix} Coronal')

    axs[2].imshow(processing_slice_to_right_orientation(data[mid_sagittal, :, :]), cmap='gray', origin=default_plt_origin_type)
    axs[2].set_title(f'{title_prefix} Sagittal')

    for ax in axs:
        ax.axis('off')

    st.pyplot(fig)