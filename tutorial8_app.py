# seg2med_app/app.py
# streamlit run tutorial8_app.py 
# F:\yang_Environments\torch\venv\Scripts\activate.ps1
# streamlit run tutorial8_app.py --server.address=0.0.0.0 --server.port=8501
# http://129.206.168.125:8501 http://169.254.3.1:8501
#import sys
#sys.path.append('./seg2med_app')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# seg2med_app/main.py

import os
import streamlit as st
import zipfile
import hashlib
import pandas as pd
import numpy as np
import nibabel as nib
from seg2med_app.simulation.get_labels import get_labels
from seg2med_app.app_utils.image_utils import (
    show_three_planes,
    show_label_overlay,
    show_three_planes_interactive,
    generate_color_map,
    load_image_canonical,
    global_slice_slider,
    image_to_base64,
    show_single_slice_image,
    show_single_slice_label,
)
from seg2med_app.ui.simulation_and_display import simulation_controls
from seg2med_app.ui.upload_and_prepare import handle_upload, compute_md5
from dataprocesser.simulation_functions import (
    _merge_seg_tissue,
    _create_body_contour_by_tissue_seg,
    _create_body_contour
)

from seg2med_app.simulation.combine_selected_organs import combine_selected_organs
from seg2med_app.ui.inference_controls import inference_controls
from seg2med_app.frankenstein.frankenstein import frankenstein_control
from seg2med_app.app_utils.titles import *
# ========== CONFIG ==========
app_root = 'seg2med_app'
os.makedirs(os.path.join(app_root, "tmp"), exist_ok=True)

# ========== UI STRUCTURE ==========
st.set_page_config(
    page_title="Frankenstein App",
    page_icon="🧠",
    layout="wide"
)


st.session_state["app_root"] = app_root

import streamlit as st
from PIL import Image
import os

def reset_app():
    st.session_state.clear()
    st.session_state.authenticated = True
    st.session_state["authenticated"] = True
    st.success("App has been reset. Login information is preserved.")
    print("App has been reset. Login information is preserved.")
    st._rerun()
    
image = Image.open(os.path.join(app_root, "frankenstein.jpg"))
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="data:image/jpeg;base64,{image_to_base64(image)}" width="150">
    </div>
    """,
    unsafe_allow_html=True
)
# copyright for logo
st.markdown(
    """
    <p style="font-size:10px; color:gray; text-align: right;">
        Logo Frankenstein – Designed by <a href="https://www.freepik.com" target="_blank" style="color:gray;">Freepik</a>
    </p>
    """,
    unsafe_allow_html=True
)


st.title("\U0001F9E0 Frankenstein - multimodal medical image generation")
st.markdown("""
**Created by**: Zeyu Yang  
PhD Student, Computer-assisted Clinical Medicine  
University of Heidelberg  

🔗 [GitHub Repository](https://github.com/musetee/frankenstein)  
📄 [Preprint on arXiv](https://arxiv.org/abs/2504.09182)  
✉️ Contact: [Zeyu.Yang@medma.uni-heidelberg.de](mailto:Zeyu.Yang@medma.uni-heidelberg.de)
""")


PASSWORD = "frankenstein"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = True # set False to be authenticated

if not st.session_state.authenticated:
    st.session_state["app_password"] = st.text_input("Enter access code", type="password")
    if st.session_state["app_password"] == PASSWORD:
        st.session_state.authenticated = True
        st.success("✅ Access granted!")
    else:
        st.warning("🔒 Please enter the correct access code to continue.")
        st.stop()

# ========== SIDEBAR (DATASET LOADER) ==========
st.sidebar.title("\U0001F9EC Dataset Loading")
load_method = st.sidebar.radio("Select load method", ["\U0001F3AE Random sample & manual draw", "\U0001F4C1 Upload segmentation"])

if st.button("🔄 Reset App"):
    reset_app

Begin = "### 🎨 Begin: Choose a colormap to visualize different tissues"

st.write(Begin)
default_cmap = "PiYG"
cmap_options = [default_cmap, "nipy_spectral", "tab20", "Set3", "Paired", "tab10", "gist_rainbow", "custom"]
selected_cmap = st.selectbox("Label colormap", cmap_options, index=0)

# 如果选择“自定义”，显示文本框供用户输入
if selected_cmap == "custom":
    custom_cmap = st.text_input("please type custom colormap name", value=default_cmap)
    selected_cmap = custom_cmap
else:
    selected_cmap = selected_cmap
    
st.session_state.update({"selected_cmap": selected_cmap})

# ========== select color map for visualization segmentation ==============
if "label_ids" in st.session_state:
    st.session_state["label_to_color"] = generate_color_map(st.session_state["label_ids"], cmap=st.session_state["selected_cmap"])
    print('organ label to color: ', list(st.session_state["label_to_color"].items())[:5])

# ========== MAIN: UPLOAD SEGMENTATION ==========
if load_method == "\U0001F4C1 Upload segmentation":
    # ========== FIRST ROW ==========
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        uploaded_file = st.file_uploader("Upload segmentation", type=["zip", "nii.gz"])
    with col2:
        uploaded_tissue = st.file_uploader("Upload tissue segmentation", type=["zip", "nii.gz"], key="tissue_upload")
    with col3:
        original_file = st.file_uploader("Upload original image", type=["nii.gz", "dcm"])
    with col4:
        # 设置 body threshold（默认值根据模态设置或用户手动输入）
        default_body_threshold = 0
        if "body_threshold" not in st.session_state:
            st.session_state["body_threshold"] = default_body_threshold
        user_input_threshold = st.number_input(
            "Body threshold for contour extraction (used on original image)",
            value=st.session_state["body_threshold"],
            step=1
        )
        use_custom_threshold = st.checkbox("Use custom body threshold", value=False)
        st.session_state["use_custom_threshold"] = use_custom_threshold
        
        if user_input_threshold:
            st.session_state["body_threshold"] = user_input_threshold
        if user_input_threshold and "orig_img" in st.session_state:
            st.session_state["contour"] = _create_body_contour(st.session_state['orig_img'], st.session_state['body_threshold'], body_mask_value=1)
            
    # ========== HASH MANAGEMENT ==========
    new_upload_hash = compute_md5(uploaded_file) if uploaded_file else None
    cached_upload_hash = st.session_state.get("uploaded_file_hash", None)
    new_tissue_hash = compute_md5(uploaded_tissue) if uploaded_tissue else None
    cached_tissue_hash = st.session_state.get("uploaded_tissue_hash", None)
    new_origin_hash = compute_md5(original_file) if original_file else None
    cached_origin_hash = st.session_state.get("uploaded_origin_hash", None)

    handle_upload(app_root, 
            uploaded_file, uploaded_tissue, original_file
        )

    # ========== SIMULATION UI (SHARED) ==========
    simulation_controls(app_root)
    
    # ========== INFERENCE UI (SHARED) ==========
    inference_controls()
    

    # ========== visualize ==========
    if "combined_seg" in st.session_state:
        z_idx, y_idx, x_idx = global_slice_slider(st.session_state["volume_shape"])
        st.session_state.update({
            "z_idx": z_idx,
            "y_idx": y_idx,
            "x_idx": x_idx,
        })
        show_three_planes_interactive(st.session_state["contour"], z_idx, y_idx, x_idx)
        show_label_overlay(st.session_state["combined_seg"], z_idx, y_idx, x_idx, label_colors=st.session_state["label_to_color"])
        
    if "selected_organs" in st.session_state and len(st.session_state["selected_organs"]) > 0:
        multi_seg = combine_selected_organs(uploaded_file)
        show_label_overlay(multi_seg, z_idx, y_idx, x_idx, label_colors=st.session_state["label_to_color"])

    if "orig_img" in st.session_state:
        show_three_planes_interactive(st.session_state["orig_img"], z_idx, y_idx, x_idx,)

    if st.session_state.get("processed_img") is not None:
        st.markdown("🔍 View Simulation Result")
        show_three_planes_interactive(st.session_state["processed_img"],
                                        st.session_state["z_idx"],
                                        st.session_state["y_idx"],
                                        st.session_state["x_idx"],)

    if st.session_state.get("output_img") is not None:
        st.session_state["output_volume_to_save"] = np.expand_dims(st.session_state["output_img"].T, axis=-1)
        show_three_planes_interactive(np.expand_dims(st.session_state["output_img"], axis=-1),
                                        0,
                                        0,
                                        0,
                                        orientation_type='none', # model output already in correct orientation
                                        )
    
                #st.success(f"Saved to {filename_output}")
        
# ========== RANDOM DRAW PAGE PLACEHOLDER ==========
elif load_method == "\U0001F3AE Random sample & manual draw":
    st.markdown("## 🎮 Frankenstein Interactive creating tool")
    frankenstein_control()
    
    
    make_step_renderer(step5_frankenstein)
    simulation_controls(app_root)
    
    make_step_renderer(step7_frankenstein)
    inference_controls()
    import matplotlib.pyplot as plt
    if "output_img" in st.session_state:
        output_img = st.session_state["output_img"]
        
        plt.figure()
        plt.imshow(output_img, cmap="gray")
        plt.grid(False)
        plt.savefig(r'seg2med_app\modeloutput.png')
        plt.close()
        width=400
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if "contour" in st.session_state:
            show_single_slice_image(st.session_state["contour"].squeeze(),title="contour")
    with col2:        
        if "combined_seg" in st.session_state:
            show_single_slice_label(st.session_state["combined_seg"].squeeze(), 
                                    st.session_state["label_to_color"], 
                                    title="combined segs")
    with col3:
        if st.session_state.get("processed_img") is not None:
            print(np.unique(st.session_state["processed_img"]))
            show_single_slice_image(st.session_state["processed_img"].squeeze(), title="image prior")

    with col4:
        if st.session_state.get("output_img") is not None:
            st.session_state["output_volume_to_save"] = np.expand_dims(st.session_state["output_img"].T, axis=-1)
            # no need to set orientation because the model output should be correct
            show_single_slice_image(st.session_state["output_img"], title="inference image", orientation_type='none')

make_step_renderer(step8_frankenstein)

# ========== SAVE ==========
output_folder = os.path.join(app_root, 'output')
os.makedirs(output_folder, exist_ok=True)
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    filename_prior = st.text_input("Filename (.nii.gz)", value="contour.nii.gz", key="filename_contour")
    prior_save_path = os.path.join(output_folder, filename_prior)

    if st.session_state.get("contour") is not None: # st.button("💾 Save Image Prior") and 
        img_to_save = nib.Nifti1Image(st.session_state["contour"], st.session_state["orig_affine"])
        nib.save(img_to_save, prior_save_path)
    if os.path.exists(prior_save_path):
        with open(prior_save_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Contour",
                data=f,
                file_name=filename_prior,
                mime="application/gzip"
            )
            #st.success(f"Saved to {filename_prior}")
with col2:
    filename_output = st.text_input("Filename (.nii.gz)", value="combined_seg.nii.gz", key="filename_combined")
    output_save_path = os.path.join(output_folder, filename_output)
    if st.session_state.get("combined_seg") is not None : # and st.button("💾 Save Output") 
        img_to_save = nib.Nifti1Image(st.session_state["combined_seg"], st.session_state["orig_affine"])
        nib.save(img_to_save, output_save_path)
    if os.path.exists(output_save_path):
        with open(output_save_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Combined Segmentation",
                data=f,
                file_name=filename_output,
                mime="application/gzip"
            )
            
with col3:
    filename_prior = st.text_input("Filename (.nii.gz)", value="prior_image.nii.gz", key="filename_prior")
    prior_save_path = os.path.join(output_folder, filename_prior)

    if st.session_state.get("processed_img") is not None: # st.button("💾 Save Image Prior") and 
        img_to_save = nib.Nifti1Image(st.session_state["processed_img"], st.session_state["orig_affine"])
        nib.save(img_to_save, prior_save_path)
    if os.path.exists(prior_save_path):
        with open(prior_save_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Prior Image",
                data=f,
                file_name=filename_prior,
                mime="application/gzip"
            )

with col4:
    filename_output = st.text_input("Filename (.nii.gz)", value="model_output.nii.gz", key="filename_output")
    output_save_path = os.path.join(output_folder, filename_output)
    if st.session_state.get("output_volume_to_save") is not None : # and st.button("💾 Save Output") 
        img_to_save = nib.Nifti1Image(st.session_state["output_volume_to_save"], st.session_state["orig_affine"])
        nib.save(img_to_save, output_save_path)
    if os.path.exists(output_save_path):
        with open(output_save_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Output Image",
                data=f,
                file_name=filename_output,
                mime="application/gzip"
            )
