# handle_upload.py
import os, zipfile, hashlib
import numpy as np
import nibabel as nib
import streamlit as st
import pandas as pd
from seg2med_app.app_utils.image_utils import load_image_canonical
from dataprocesser.simulation_functions import _merge_seg_tissue, _create_body_contour_by_tissue_seg, _create_body_contour
from seg2med_app.simulation.get_labels import get_labels
from seg2med_app.app_utils.image_utils import global_slice_slider, show_three_planes_interactive, show_label_overlay, generate_color_map
def compute_md5(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

def handle_upload(app_root, uploaded_file, uploaded_tissue, original_file):
    if uploaded_file is None or uploaded_tissue is None:
        return

    tmpdir = os.path.join(app_root, "tmp")
    os.makedirs(tmpdir, exist_ok=True)

    # 哈希判断是否变化
    file_hash = compute_md5(uploaded_file)
    tissue_hash = compute_md5(uploaded_tissue)
    origin_hash = compute_md5(original_file) if original_file else None

    if file_hash == st.session_state.get("uploaded_file_hash") and \
       tissue_hash == st.session_state.get("uploaded_tissue_hash") and\
        origin_hash == st.session_state.get("uploaded_origin_hash"):
        return  # 未变化则不重复处理

    # 保存上传文件
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    tissue_path = os.path.join(tmpdir, uploaded_tissue.name)
    with open(tissue_path, "wb") as f:
        f.write(uploaded_tissue.read())

    # 解压
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        with zipfile.ZipFile(tissue_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        nii_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".nii.gz")]

        seg_max = len(nii_files)
        organ_to_label, label_to_organ, label_ids, input_modality = get_labels(seg_max, app_root)

        ref_img = load_image_canonical(nii_files[0])
        shape = ref_img.shape
        combined = np.zeros(shape, dtype=np.uint8)
        organ_names, empty_organs, organ_paths = [], [], {}

        for path in nii_files:
            name = os.path.basename(path).replace(".nii.gz", "")
            if name in organ_to_label:
                label = organ_to_label[name]
                organ_paths[name] = path
                mask = load_image_canonical(path).get_fdata()
                if np.all(mask == 0):
                    empty_organs.append(name)
                else:
                    organ_names.append(name)
                    combined[mask > 0] = label
        contour = _create_body_contour_by_tissue_seg(combined, body_mask_value=1, area_threshold=500)
        
    else:
        ref_img = load_image_canonical(file_path)
        ref_data = ref_img.get_fdata()
        tissue_img = load_image_canonical(tissue_path)
        tissue_data = tissue_img.get_fdata()

        contour = _create_body_contour_by_tissue_seg(tissue_data, body_mask_value=1, area_threshold=500)
        combined = _merge_seg_tissue(ref_data, tissue_data)

        label_ids_in_volume = np.unique(combined).astype(int)
        seg_max = np.max(ref_data)
        organ_to_label, label_to_organ, label_ids, input_modality = get_labels(seg_max, app_root)
        organ_names = [label_to_organ.get(i) for i in label_ids_in_volume if i in label_to_organ]
        print('available organs:', organ_names)
        print('corresponding labels:', label_ids_in_volume)
        organ_paths = {}

    # 原图
    if original_file:
        orig_path = os.path.join(tmpdir, original_file.name)
        with open(orig_path, "wb") as f:
            f.write(original_file.read())

        if original_file.name.endswith(".nii.gz"):
            orig_img_obj = load_image_canonical(orig_path)
            orig_img = orig_img_obj.get_fdata()
            orig_affine = orig_img_obj.affine
        else:
            orig_img, orig_affine = combined, ref_img.affine
        
        if not st.session_state["use_custom_threshold"]:
            modality_default_threshold = {
                "ct": -500,
                "mr": 50, 
                "T1_GRE": 100,
                "T2_SPACE": 10,
                "T1_VIBE_IN": 30,
                "T1_VIBE_OPP": 30,
                "T1_VIBE_DIXON": 30
            }
            st.session_state["body_threshold"] = modality_default_threshold.get(input_modality, 0)
        
        contour = _create_body_contour(orig_img, st.session_state['body_threshold'], body_mask_value=1)
        print('create body contour using threshold ', st.session_state['body_threshold'])
        st.session_state["orig_img"] = orig_img
    else:
        orig_affine = ref_img.affine
        print("no original file, pay attention")

    label_to_color = generate_color_map(label_ids, cmap=st.session_state["selected_cmap"])

    # 保存状态
    st.session_state.update({
        "combined_seg": combined,
        "organ_names": organ_names,
        "reference_affine": ref_img.affine,
        "volume_shape": combined.shape,
        "label_ids": label_ids,
        "organ_to_label": organ_to_label,
        "valid_organ_paths": organ_paths,
        "contour": contour,
        "input_modality": input_modality,
        "orig_affine": orig_affine,
        "uploaded_file_hash": file_hash,
        "uploaded_tissue_hash": tissue_hash,
        "uploaded_origin_hash": origin_hash,
        "label_to_color": label_to_color, 
    })

    
    