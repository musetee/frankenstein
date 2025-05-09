# seg2med_app/app.py
# streamlit run tutorial8_app.py
#import sys
#sys.path.append('./seg2med_app')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import zipfile
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seg2med_app.simulation.simulator import simulate_modality
from seg2med_app.simulation.get_labels import get_labels
from seg2med_app.app_utils.image_utils import show_three_planes, show_label_overlay, show_three_planes_interactive, generate_color_map, load_image_canonical
from dataprocesser.simulation_functions import _merge_seg_tissue, _create_body_contour_by_tissue_seg
import hashlib

def compute_md5(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)  # 复位文件指针
    return file_hash

app_root = 'seg2med_app'

#st.set_page_config(layout="wide")


st.set_page_config(layout="wide")
st.title("🧠 seg2med multimodal medical image generation")
st.sidebar.title("🧬 Dataset Loading")

load_method = st.sidebar.radio("Select load method", ["📁 Upload segmentation", "🎮 Random sample & manual draw"])
if load_method == "📁 Upload segmentation":
    st.title("📁 Upload your segmentation")

    # 上传分割图（支持 zip 或单个 nii.gz）
    st.markdown("### 📤 upload segmentation")
    uploaded_file = st.file_uploader("upload .zip or .nii.gz segmentation", type=["zip", "nii.gz"])

    # === Upload extra tissue segmentations ===
    st.markdown("### 🧩 upload basic tissue segmentation")
    uploaded_tissue = st.file_uploader("Upload zip file with subcutaneous_fat, torso_fat, skeletal_muscle", type=["zip", "nii.gz"], key="tissue_zip")

    # 📌 上传原图（支持 .nii.gz 或 单张 .dcm）
    st.markdown("### 🖼️ upload original image (optional)")
    original_file = st.file_uploader("upload original iamge NIfTI or singal DICOM", type=["nii.gz", "dcm"])

    # 计算文件哈希（更稳定）
    new_upload_hash = compute_md5(uploaded_file) if uploaded_file else None
    cached_upload_hash = st.session_state.get("uploaded_file_hash", None)

    new_tissue_hash = compute_md5(uploaded_tissue) if uploaded_tissue else None
    cached_tissue_hash = st.session_state.get("uploaded_tissue_hash", None)

    new_origin_hash = compute_md5(original_file) if original_file else None
    cached_origin_hash = st.session_state.get("uploaded_origin_hash", None)

    if "contour" not in st.session_state:
        st.session_state["contour"] = None
        st.session_state["processed_img"] = None

    # 预定义 mapping
    tissue_label_map = {
        "subcutaneous_fat": 201,
        "torso_fat": 202,
        "skeletal_muscle": 203,
    }

    tmpdir_seg = os.path.join(app_root, "tmp")
    os.makedirs(tmpdir_seg, exist_ok=True)  # 若不存在则创建


    if st.button("⚙️ clear cache"):
        for f in os.listdir(tmpdir_seg):
            os.remove(os.path.join(tmpdir_seg, f))

    if uploaded_file is not None and uploaded_tissue is not None and new_upload_hash != cached_upload_hash:
        input_path = os.path.join(tmpdir_seg, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        input_path_tissue = os.path.join(tmpdir_seg, uploaded_tissue.name)
        with open(input_path_tissue, "wb") as f:
            f.write(uploaded_tissue.read())

        # 解压或直接读取
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir_seg)
            
            with zipfile.ZipFile(input_path_tissue, 'r') as zip_ref:
                zip_ref.extractall(tmpdir_seg)
            
            nii_files = [os.path.join(tmpdir_seg, f) for f in os.listdir(tmpdir_seg) if f.endswith((".nii.gz", ".nii"))]
            tissue_files = [os.path.join(tmpdir_seg, f) for f in os.listdir(tmpdir_seg) if f.endswith((".nii.gz", ".nii"))]

            st.success(f"unzip and get {len(nii_files)} segmentations")

            seg_max = len(nii_files)
            organ_to_label, label_to_organ, label_ids, input_modality = get_labels(seg_max, app_root)

            # 构建组合分割图像 volume（初始化）
            st.info("combine all segmentations into one...")
            reference_img = load_image_canonical(nii_files[0])
            volume_shape = reference_img.shape
            combined_seg = np.zeros(volume_shape, dtype=np.uint8)
            organ_names = []
            empty_organ_names = []
            valid_organ_paths = {}
            for path in nii_files:
                organ_name = os.path.basename(path).replace(".nii.gz", "")
                if organ_name in organ_to_label:
                    label_value = organ_to_label[organ_name]
                    valid_organ_paths[organ_name] = path
                    mask = load_image_canonical(path).get_fdata()
                    if np.all(mask == 0):
                        empty_organ_names.append(organ_name)
                    else:
                        organ_names.append(organ_name)
                    combined_seg[mask > 0] = label_value
                else:
                    st.warning(f"unrecognized organs: {organ_name}")
            if len(empty_organ_names)>0:
                st.warning(f"Empty segmentation: {empty_organ_names}")
            st.session_state["valid_organ_paths"] = valid_organ_paths
        else:
            reference_img = load_image_canonical(input_path)
            seg = reference_img.get_fdata()
            print('seg shape:', seg.shape)
            print('seg min max:', np.min(seg), np.max(seg))
            seg_tissue_nii_file = load_image_canonical(input_path_tissue)
            seg_tissue = seg_tissue_nii_file.get_fdata()

            contour = _create_body_contour_by_tissue_seg(seg_tissue, body_mask_value=1, area_threshold=500)

            combined_seg = _merge_seg_tissue(seg, seg_tissue)

            volume_shape = seg.shape
            seg_max = np.max(seg)
            organ_to_label, label_to_organ, label_ids, input_modality = get_labels(seg_max, app_root)

            label_ids_in_volume = np.unique(combined_seg).astype(int)
            organ_names = []
            unrecognized_labels =[]
            for label_id in label_ids_in_volume:
                if label_id in label_to_organ:
                    organ_names.append(label_to_organ[label_id])
                elif label_id != 0:
                    unrecognized_labels.append(label_id)

            st.warning(f"Label {unrecognized_labels} not recognized in label map, but no influence to results")

        st.success(f"sucessfully load segmentation volume, shape: {volume_shape}")    
        st.session_state["reference_affine"] = reference_img.affine
        st.session_state["volume_shape"] = volume_shape
        st.session_state["organ_to_label"] = organ_to_label
        st.session_state["label_ids"] = label_ids
        st.session_state["combined_seg"] = combined_seg
        st.session_state["organ_names"] = organ_names
        st.session_state["uploaded_file_hash"] = new_upload_hash
        st.session_state["uploaded_tissue_hash"] = new_tissue_hash
        st.session_state["contour"] = contour
        st.session_state["processed_img"] = None
        st.session_state["input_modality"] = input_modality

    if uploaded_file is not None and uploaded_tissue is not None:
        combined_seg = st.session_state["combined_seg"]
        organ_names = st.session_state["organ_names"]
        contour = st.session_state["contour"] 
        volume_shape = st.session_state["volume_shape"]
        organ_to_label = st.session_state["organ_to_label"]
        label_ids = st.session_state["label_ids"] 
        input_modality = st.session_state["input_modality"]

        reference_img = nib.Nifti1Image(combined_seg, affine=st.session_state["reference_affine"])

        cmap_options = ["nipy_spectral", "tab20", "Set3", "Paired", "tab10", "gist_rainbow"]
        selected_cmap = st.selectbox("🎨 Select colormap for label display", cmap_options, index=0)
        label_to_color = generate_color_map(label_ids, cmap=selected_cmap)
        st.success(f"sucessfully combined {len(organ_names)} organs")

        show_label_overlay(combined_seg, title="combined segmentations (with color)", label_colors=label_to_color, view_id='10')
        show_three_planes_interactive(contour, view_id='03')
        # 多器官选择并显示组合
        st.markdown("### 🧩 Select an organ to view the corresponding segmentation (multiple selections are allowed)")
        selected_organs = st.multiselect("Select one or more organs", organ_names)
        if selected_organs:
            if uploaded_file.name.endswith(".zip"):
                valid_organ_paths = st.session_state["valid_organ_paths"]
                multi_seg = np.zeros(volume_shape, dtype=np.uint8)
                for organ_name in selected_organs:
                    label_value = organ_to_label[organ_name]
                    mask = load_image_canonical(valid_organ_paths[organ_name]).get_fdata()
                    multi_seg[mask > 0] = label_value
            elif uploaded_file.name.endswith((".nii.gz", ".nii")):
                multi_seg = np.zeros(volume_shape, dtype=np.uint8)
                for organ_name in selected_organs:
                    label_value = organ_to_label[organ_name]
                    multi_seg[combined_seg == label_value] = label_value
            show_label_overlay(multi_seg, title="Selected organ segmentation", label_colors=label_to_color, view_id='11')
        # 如果用户上传了原图，也展示一下
        # 通用处理流程
        '''if target_data is not None:
            contour = _create_body_contour(target_data, input_config['body_threshold'], body_mask_value=1)
        else:
            contour = _create_body_contour_by_tissue_seg()'''
        if original_file:
            if new_origin_hash != cached_origin_hash:
                st.markdown("### 🖼️ original")
                orig_path = os.path.join(tmpdir_seg, original_file.name)
                with open(orig_path, "wb") as f:
                    f.write(original_file.read())
                
                if original_file.name.endswith(".nii.gz"):
                    orig_data = load_image_canonical(orig_path)
                    orig_img = orig_data.get_fdata()
                    orig_affine = orig_data.affine
                    st.session_state["orig_img"] = orig_img
                    st.session_state["orig_affine"] = orig_affine
                    st.info(orig_affine)
                elif original_file.name.endswith(".dcm"):
                    import pydicom
                    ds = pydicom.dcmread(orig_path)
                    dcm_img = ds.pixel_array
                    st.image(dcm_img, caption="original image DICOM", clamp=True, use_column_width=True)
                st.session_state["uploaded_origin_hash"] = new_origin_hash
        
        else:
            #orig_img = combined_seg
            #st.session_state["orig_img"] = orig_img
            st.session_state["orig_affine"] = st.session_state["reference_affine"]

        
        if "orig_img" in st.session_state:
            show_three_planes_interactive(st.session_state["orig_img"], title_prefix="original image", view_id='02')
        
        # 模态选择
        modality_options = ["CT", "T1_GRE", "T2_SPACE", "T1_VIBE_IN", "T1_VIBE_OPP", "T1_VIBE_DIXON"]
        modality = st.selectbox("🧬 select modality", modality_options)

        # 创建对应索引（0 到 5）
        modality_idx = modality_options.index(modality)

        # 自动切换不同参数表（根据模态和组织类型）
        # 👉 TODO: 在 simulation/ 目录下准备以下两个文件（你后续会补充）
        # simulation/params_ct.csv
        # simulation/params_mr.csv

        st.markdown("### 🧾 simulation value table")
        param_file = st.file_uploader("you can upload your own tables", type="csv")
        params_csv_ct = os.path.join(app_root, "simulation/params_ct.csv")
        params_csv_mr = os.path.join(app_root, "simulation/params_mr.csv")
        if param_file:
            df_params = pd.read_csv(param_file)
        else:
            if modality == "CT":
                df_params = pd.read_csv(params_csv_ct)  
            else:
                df_params = pd.read_csv(params_csv_mr)  
        st.dataframe(df_params, use_container_width=True)

elif load_method == "🎮 Random sample & manual draw":
    st.title("🎮 Select or draw from predefined dataset")

if st.button("⚙️ run simulation"):
    st.info("run simulation...")
    
    print(st.session_state["orig_affine"])
    processed_img = simulate_modality(contour, modality_idx, combined_seg, input_modality, params_csv_ct, params_csv_mr)
    
    st.success("simulation finished ✅")
    st.markdown("### 🔍 view")

    st.session_state["processed_img"] = processed_img
    print("simulation finished")

    

if st.session_state["processed_img"] is not None:
    show_three_planes_interactive(st.session_state["processed_img"], view_id='04')

'''if st.button("⚙️ save prior image"):
    output_folder = os.path.join(app_root, 'output')
    patient_ID = 'example.nii.gz'
    output_dir = os.path.join(output_folder,patient_ID)
    os.makedirs(output_dir,exist_ok=True)
    prior_to_save = nib.Nifti1Image(st.session_state["processed_img"], st.session_state["orig_affine"])
    nib.save(prior_to_save, output_dir)'''
col1, col2 = st.columns([3, 1])
with col1:
    filename = st.text_input("Filename (.nii.gz)", value="example.nii.gz")

with col2:
    save_button = st.button("💾 Save")

if save_button:
    output_folder = os.path.join(app_root, 'output')
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, filename)

    # 保存 NIfTI
    prior_to_save = nib.Nifti1Image(st.session_state["processed_img"], st.session_state["orig_affine"])
    nib.save(prior_to_save, save_path)

    st.success(f"Image saved to {save_path}")

if st.button("⚙️ run reference"):
    from synthrad_conversion import train
    train.run(config='tutorial2_train_ablation_3.yaml', 
              dataset_name = 'multimodal_prior_csv',
              data_dir = r'E:\Projects\yang_proj\data\seg2med')