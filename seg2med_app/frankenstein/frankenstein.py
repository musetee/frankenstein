import os

import nibabel as nib
import numpy as np
import streamlit as st
from PIL import Image

from seg2med_app.streamlit_drawable_canvas import st_canvas
from seg2med_app.simulation.get_labels import get_labels
from seg2med_app.app_utils.image_utils import show_three_planes_interactive, show_label_overlay
import time
from tqdm import tqdm
from seg2med_app.frankenstein.utils import *
from seg2med_app.frankenstein.random_load import *
# ----------------------------
# 随机加载函数
# ----------------------------
import os
import numpy as np
import nibabel as nib

from seg2med_app.app_utils.image_utils import processing_slice_to_right_orientation, restore_slice_to_wrong_orientation
from seg2med_app.app_utils.titles import *

# ----------------------------
# 显示切片函数
# ----------------------------
import matplotlib.pyplot as plt

def frankenstein_control():
    def reset_canvas():
        print('reset canvas')
        st.session_state["result_json_data_objects"] = None
        st.session_state["canvas_initialized"] = False
        st.session_state["canvas_key"] = f"canvas_{int(time.time())}"
        
    samples_path = os.path.join(st.session_state["app_root"], "samples")
    needed_organs = [
        'spleen', 'kidney_right', 'kidney_left', 'liver', 'stomach', 'pancreas', 
        'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left', 
        'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 
        'lung_lower_lobe_right', 'esophagus', 'small_bowel', 
        'duodenum', 'colon', 'vertebrae_L5', 'vertebrae_L4', 
        'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 
        'vertebrae_T11', 'vertebrae_T10', 'heart', 'aorta', 
        'inferior_vena_cava', 'portal_vein_and_splenic_vein', 
        'iliac_artery_left', 'iliac_artery_right', 
        'iliac_vena_left', 'iliac_vena_right', 'hip_left', 'hip_right', 
        'spinal_cord', 'gluteus_medius_left', 'autochthon_left', 'autochthon_right', 
        'iliopsoas_left', 'iliopsoas_right', 'rib_left_5', 'rib_left_6', 
        'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 
        'rib_left_12', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 
        'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12', 
        'sternum', 'costal_cartilages',
    ]
    
    default_flexible_organs = ['spleen', 'liver', 'stomach','aorta', 'heart', 'inferior_vena_cava']
    
    #default create CT segmentation
    seg_max = 100 
    organ_to_label, label_to_organ, label_ids, input_modality = get_labels(seg_max, st.session_state["app_root"])
    st.session_state.update({
        "label_ids": label_ids,
        "organ_to_label": organ_to_label,
        "input_modality": input_modality,
        "label_to_organ": label_to_organ,
    })
    
    # ==================step2====================
    make_step_renderer(step1_frankenstein)
    # multiselect module cannot be placed in front of button! otherwise button will be triggered
    st.session_state["selected_organs_frankenstein"] = list(st.multiselect("Organs to be edited:",needed_organs, default=default_flexible_organs))# default_flexible_organs 
    st.session_state["fixed_organs_frankenstein"]  =  list(set(needed_organs) - set(st.session_state["selected_organs_frankenstein"] ))
    print("selected_organs_frankenstein - ",st.session_state["selected_organs_frankenstein"] )
    print("fixed_organs_frankenstein - ",st.session_state["fixed_organs_frankenstein"])
    
    # ----------------------------
    # 主程序
    # ----------------------------
    #st.title("🎮🧠 Frankenstein Module")
    
    if "existed_frankenstein_hash" not in st.session_state:
        st.session_state["existed_frankenstein_hash"] = ""
        st.session_state["current_frankenstein_hash"] = ""
    
    # ==================step2====================
    make_step_renderer(step2_frankenstein)
    
    if st.button("🧍 Default Mode"):
        print("click random load organs button")
        st.session_state["current_frankenstein_hash"] = hash_current_time()
        (st.session_state["contour_frankenstein"], 
        st.session_state["organs_frankenstein"], 
        st.session_state["seg_tissue_frankenstein"],
        st.session_state["organ_source_mapping"], 
        st.session_state["minimum_fixed_organ_slice_num"]) = random_load_chaotic(samples_path, 
                                                                        st.session_state["selected_organs_frankenstein"], 
                                                                        st.session_state["fixed_organs_frankenstein"],
                                                                        'default')
        
        '''(st.session_state["contour_frankenstein"], 
        st.session_state["organs_frankenstein"], 
        st.session_state["seg_tissue_frankenstein"]) = random_load(samples_path, needed_organs)'''
        
    elif st.button("🧪 Semi-Chaotic Mode (recommended)"):
        print("click random load organs button")
        st.session_state["current_frankenstein_hash"] = hash_current_time()
        (st.session_state["contour_frankenstein"], 
        st.session_state["organs_frankenstein"], 
        st.session_state["seg_tissue_frankenstein"],
        st.session_state["organ_source_mapping"], 
        st.session_state["minimum_fixed_organ_slice_num"]) = random_load_chaotic(samples_path, 
                                            st.session_state["selected_organs_frankenstein"], 
                                            st.session_state["fixed_organs_frankenstein"],
                                            'semi-chaotic')
    
    elif st.button("💥 fully-chaotic Mode"):
        print("click random load organs button")
        st.session_state["current_frankenstein_hash"] = hash_current_time()
        (st.session_state["contour_frankenstein"], 
        st.session_state["organs_frankenstein"], 
        st.session_state["seg_tissue_frankenstein"],
        st.session_state["organ_source_mapping"], 
        st.session_state["minimum_fixed_organ_slice_num"]) = random_load_chaotic(samples_path, 
                                            st.session_state["selected_organs_frankenstein"], 
                                            st.session_state["fixed_organs_frankenstein"],
                                            'fully-chaotic')
    
    #### descriptions
    import pandas as pd

    if "organ_source_mapping" in st.session_state:
        with st.expander("📋 Organ Source Summary (click to expand)"):
            df_sources = pd.DataFrame([
                {"Organ": organ, "Patient ID": pid}
                for organ, pid in st.session_state["organ_source_mapping"].items()
            ])
            st.dataframe(df_sources, use_container_width=True)
    from seg2med_app.frankenstein.infos import modes_info
    modes_info()
    
    print('existed_frankenstein_hash:', st.session_state["existed_frankenstein_hash"])
    print('current_frankenstein_hash:', st.session_state["current_frankenstein_hash"])
    
    
    if st.session_state["current_frankenstein_hash"] != st.session_state["existed_frankenstein_hash"]:
        st.session_state["existed_frankenstein_hash"] = st.session_state["current_frankenstein_hash"]
        #slider = st.slider(f"{organ_name} slice", 0, num_slices - 1, num_slices // 2)
        
        progress_text = "⏳ processing organ data..."
        progress_bar = st.progress(0, text=progress_text)
        total_steps = len(st.session_state["selected_organs_frankenstein"]) + len(st.session_state["fixed_organs_frankenstein"])
        step = 0
        
        for organ in tqdm(st.session_state["selected_organs_frankenstein"]):
            organ_nii = st.session_state["organs_frankenstein"][organ]
            organ_nii_data = organ_nii
            key_organ_slice = f"slice_frankenstein_{organ}"
            key_organ = f"{organ}_frankenstein"
            st.session_state[key_organ] = organ_nii #get_slice_img(organ_nii.get_fdata().astype(np.uint8),organ)
            step += 1
            progress_bar.progress(step / total_steps, text=f"{progress_text} {step / total_steps}")

        first_fixed_organ = st.session_state["fixed_organs_frankenstein"][0]
        for organ in tqdm(st.session_state["fixed_organs_frankenstein"]):
            organ_nii = st.session_state["organs_frankenstein"][organ]
            organ_nii_data = organ_nii
            #key_organ_slice = f"slice_frankenstein_{organ}"
            key_organ = f"{organ}_frankenstein"
            st.session_state[key_organ] = organ_nii_data # get_slice_img(organ_nii_data,organ)
            step += 1
            progress_bar.progress(step / total_steps, text=f"{progress_text} {step} / {total_steps}")
            
        progress_bar.progress(1.0, text="✅ processing finished!")
    
    # ****initialize slice indices hash
    if "prev_slice_indices_hash" not in st.session_state:
        st.session_state["prev_slice_indices_hash"] = ""
    
    
    if "contour_frankenstein" in st.session_state:
        # ==================step3====================
        make_step_renderer(step3_frankenstein)
        # st.write("### Select organ slices")

        #### processing for visualization:
        #### mask volume ---slider----> mask slice --> organ label --> rgb map
        
        ###### processing contour volume
        contour_frankenstein_volume = st.session_state["contour_frankenstein"]
        seg_tissue_frankenstein_volume = st.session_state["seg_tissue_frankenstein"]
        seg_tissue_frankenstein_volume = seg_tissue_frankenstein_volume.astype(np.uint8)
        seg_tissue_frankenstein_volume[seg_tissue_frankenstein_volume == 1] = 201 # tombine contour here
        seg_tissue_frankenstein_volume[seg_tissue_frankenstein_volume == 2] = 202
        seg_tissue_frankenstein_volume[seg_tissue_frankenstein_volume == 3] = 203
        
        
        contour_slice_num = st.session_state["contour_frankenstein"].shape[2]
        seg_tissue_slider_num = st.session_state["contour_frankenstein"].shape[2]
        fixed_organ_slider_num = st.session_state["minimum_fixed_organ_slice_num"]
        col1, col2, col3 = st.columns(3)
        with col1:
            contour_slice_idx = st.slider("select index for contour", 0, contour_slice_num-1, contour_slice_num//2, key = 'slice_idx_contour')
        with col2:
            seg_tissue_slice_idx = st.slider("select index for seg tissue", 0, seg_tissue_slider_num-1, seg_tissue_slider_num//2, key = 'slice_idx_seg_tissue')
        with col3:
            fixed_orgna_slice_idx = st.slider("select index for fixed organs", 0, fixed_organ_slider_num-1, fixed_organ_slider_num//2, key = 'slice_idx_fixed_organ')
        ###### processing fixed organs, combine with tissue_seg
        st.session_state["slice_frankenstein_contour"] = processing_slice_to_right_orientation(contour_frankenstein_volume[:,:,contour_slice_idx])
        st.session_state["slice_frankenstein_seg_tissue"] = processing_slice_to_right_orientation(seg_tissue_frankenstein_volume[:,:,seg_tissue_slice_idx])
        
        combine_label_fixed_organ_slice = np.zeros_like(processing_slice_to_right_orientation(seg_tissue_frankenstein_volume[:,:,fixed_orgna_slice_idx]))
        for fixed_organ_name in st.session_state["fixed_organs_frankenstein"]:
            key_organ = f"{fixed_organ_name}_frankenstein"
            fixed_organ = st.session_state[key_organ]
            fixed_organ_volume = fixed_organ 
            fixed_organ_slice = processing_slice_to_right_orientation(fixed_organ_volume[:,:,fixed_orgna_slice_idx])
            label_mask_fixed_organ = convert_organ_mask_to_label(fixed_organ_slice, fixed_organ_name, st.session_state["organ_to_label"])
            combine_label_fixed_organ_slice += label_mask_fixed_organ
        combine_fixed_organ_rgb_map = convert_label_to_rgb_map(combine_label_fixed_organ_slice, st.session_state["label_to_color"])
        
        
        debug_volume_orientation=False
        if debug_volume_orientation:
            show_three_planes_interactive(seg_tissue_frankenstein_volume, seg_tissue_slice_idx, 10, 10)
            show_label_overlay(seg_tissue_frankenstein_volume, seg_tissue_slice_idx, 10, 10,st.session_state["label_to_color"])
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            label_mask_contour, rgb_contour = show_organ_axial(st.session_state["slice_frankenstein_contour"],'contour',st.session_state["label_to_color"],st.session_state["organ_to_label"],)
        with col2:
            label_mask_seg_tissue, rgb_seg_tixssue = show_organ_axial(st.session_state["slice_frankenstein_seg_tissue"],'seg_tissue',st.session_state["label_to_color"],st.session_state["organ_to_label"],)
        with col3:
            st.image(Image.fromarray(combine_fixed_organ_rgb_map), caption=f"fixed organ", use_column_width=True)
        
        ###### processing selected organs for transformation
        selected_organs = st.session_state["selected_organs_frankenstein"]
        n_cols = 6  # 每行显示几个器官
        cols = st.columns(n_cols)

        for i, organ_name in enumerate(selected_organs):
            key_volume = f"{organ_name}_frankenstein"
            organ_volume = st.session_state[key_volume]

            slider_key = f"slice_idx_{organ_name}"
            slice_idx_default = organ_volume.shape[2] // 2
            col = cols[i % n_cols]  
            with col:
                st.slider(
                    f"{organ_name}", 
                    0, organ_volume.shape[2] - 1,
                    value=slice_idx_default,
                    key=slider_key,
                )
                slice_data = processing_slice_to_right_orientation(organ_volume[:, :, st.session_state[slider_key]])
                label_mask = convert_organ_mask_to_label(slice_data, organ_name, st.session_state["organ_to_label"])
                rgb_img = convert_label_to_rgb_map(label_mask, st.session_state["label_to_color"])
                st.image(Image.fromarray(rgb_img), use_column_width=True)
                
                key_organ = f"slice_frankenstein_{organ_name}"
                st.session_state[key_organ] = slice_data
        
        slice_indices_state = {
            "fixed": st.session_state["slice_idx_fixed_organ"]
        }
        
        for organ in st.session_state["selected_organs_frankenstein"]:
            slider_key = f"slice_idx_{organ}"
            slice_indices_state[organ] = st.session_state[slider_key]

        current_hash = hash_slice_indices(slice_indices_state)
        
        # ==================step4====================
        make_step_renderer(step4_frankenstein)
        st.write("### ✍️ Organ canvas")
        st.session_state["editable_organ"] = st.session_state["selected_organs_frankenstein"] + ["contour", "seg_tissue"]
        
        '''st.multiselect("✍️ Current editable organs",
            st.session_state["selected_organs_frankenstein"] + ["contour", "seg_tissue"],
            default = st.session_state["selected_organs_frankenstein"]
            #key="editable_organ"
        )'''
        
        #### convert mask to path object using opencv
        #### and add to canvas backend
        if current_hash != st.session_state["prev_slice_indices_hash"]:
            st.session_state["prev_slice_indices_hash"] = current_hash
            print("🔄 Slice indices changed, reloading slices...")

            drawing_objects = []
        
            # 添加 contour
            if "slice_frankenstein_contour" in st.session_state:
                is_editable = "contour" in st.session_state["editable_organ"]
                contour_mask = st.session_state["slice_frankenstein_contour"] > 0
                drawing_objects.extend(mask_to_path_object(contour_mask, "rgba(255,255,255,0.3)", "contour", selectable=is_editable))

            # interactive seg_tissue
            if "slice_frankenstein_seg_tissue" in st.session_state:
                # for tissue seg: 201=subcutaneous_fat, 202=torso_fat, 203=skeletal_muscle
                is_editable_seg_tissue = "seg_tissue" in st.session_state["editable_organ"]
                print(np.unique(st.session_state["slice_frankenstein_seg_tissue"]))
                tissue_label_map = {
                    201: "subcutaneous_fat",
                    202: "torso_fat",
                    203: "skeletal_muscle"
                }

                for label_value, name in tissue_label_map.items():
                    mask = st.session_state["slice_frankenstein_seg_tissue"] == label_value
                    if np.sum(mask) == 0:
                        continue
                    rgb = st.session_state["label_to_color"].get(label_value, "255,0,0")
                    rgba = f"rgba({rgb}, 0.4)"
                    #drawing_objects.extend(mask_to_path_object(mask, rgba, name, selectable=is_editable_seg_tissue))
            
            print("editable organs:", st.session_state["editable_organ"])
            
            for organ in st.session_state["selected_organs_frankenstein"]:
                key_organ = f"slice_frankenstein_{organ}"
                mask = st.session_state[key_organ]
                if np.sum(mask) == 0:
                    continue
                label = st.session_state["organ_to_label"][organ]
                rgb_str = st.session_state["label_to_color"].get(label, "255,0,0")
                rgba = f"rgba({rgb_str}, 0.4)"
                is_editable = organ in st.session_state["editable_organ"]
                drawing_objects.extend(mask_to_path_object(mask, rgba, organ, selectable=is_editable))

            # set seg tissue as background
            try:
                all_fixed = combine_fixed_organ_rgb_map+rgb_seg_tixssue
                st.session_state["tissue_rgb"] = Image.fromarray(all_fixed) #seg_tissue_to_rgb_image(st.session_state["slice_frankenstein_seg_tissue"], st.session_state["label_to_color"])
            except:
                st.session_state["tissue_rgb"] = Image.fromarray(np.zeros(st.session_state["contour_frankenstein"].shape[:2], dtype=np.uint8))
            
            st.session_state["initial_drawing"] = {"objects": drawing_objects} # st.session_state["result_json_data_objects"]
        else:
            print("✅ Slice indices unchanged, skip reload.")
        
        canvas_result = st_canvas(
            fill_color=None,
            stroke_width=2,
            background_image=st.session_state["tissue_rgb"], # Image.fromarray(np.zeros(st.session_state["contour_frankenstein"].shape[:2], dtype=np.uint8))
            update_streamlit=True,
            height=st.session_state["contour_frankenstein"].shape[0],
            width=st.session_state["contour_frankenstein"].shape[1],
            drawing_mode="transform",
            initial_drawing=st.session_state["initial_drawing"],
            key=st.session_state.get("canvas_key", "canvas_default")
        )
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            st.session_state["result_json_data_objects"] = canvas_result.json_data["objects"]
            
            canvas_shape = st.session_state["contour_frankenstein"].shape[:2]  # 高宽

            editable_label_mask, contour_label_mask, editable_organ_label_mask = canvas_json_to_label_mask(
                st.session_state["result_json_data_objects"],
                canvas_shape,
                st.session_state["label_to_color"],
                st.session_state["organ_to_label"]
            )
            
            all_organ_label_mask = np.maximum(combine_label_fixed_organ_slice, editable_organ_label_mask)
            all_organ_label_mask = np.maximum(all_organ_label_mask, st.session_state["slice_frankenstein_seg_tissue"])
            
            all_label_mask_with_modified_editable = contour_label_mask + all_organ_label_mask
            
            
            print('values in all_organ_label_mask', np.unique(all_organ_label_mask))
            print('values in contour_label_mask', np.unique(contour_label_mask))
            print('label mask value:', np.unique(editable_label_mask))
            
            st.write("🧪 transformed organ Label Map")
            
            gray_rgb = normalize_label_mask_log_enhanced(all_label_mask_with_modified_editable)
            st.image(gray_rgb, caption="Canvas-based Label Map")
                        
            # 保存到 session 或用于导出
            combined_seg_for_inference = restore_slice_to_wrong_orientation(all_organ_label_mask)
            contour_for_inference = restore_slice_to_wrong_orientation(contour_label_mask)
            
            st.session_state["canvas_label_mask"] = all_label_mask_with_modified_editable
            st.session_state.update({
                "combined_seg": np.expand_dims(combined_seg_for_inference,-1).astype(np.int32),
                "contour": np.expand_dims(contour_for_inference,-1).astype(np.int32),
                "volume_shape": (contour_label_mask.shape[0],contour_label_mask.shape[1],1),
                "input_modality": "ct",
                "z_idx": 0,
                
            })
            
            
        if st.button("🧹 reset canvas"):
            reset_canvas()

        '''
        with open(os.path.join(st.session_state["app_root"], "frankenstein","organ_editor.html"), "r", encoding="utf-8") as f:
            html_template = f.read()
        fabric_data_json = json.dumps({"objects": drawing_objects})
        # 注入 JSON 到 HTML 中
        canvas_size = 512
        st.session_state["html_filled"] = html_template.replace("{{ data }}", fabric_data_json)
        components.html(st.session_state["html_filled"], height=canvas_size+100, width=canvas_size+100)

        '''
        

        #if canvas_result.json_data is not None:
        #    st.write("Canvas 编辑 JSON:", canvas_result.json_data)

        '''if st.button("导出空白图像"):
            fake_output = np.zeros_like(st.session_state["contour_frankenstein"].get_fdata()[:, :, slice_idx])
            nii_file = save_custom_slice(fake_output, st.session_state["contour_frankenstein"].affine)
            st.download_button("下载 NIfTI 文件", nii_file, file_name="custom_seg.nii.gz")'''
