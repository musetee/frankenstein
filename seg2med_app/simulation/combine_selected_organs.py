import streamlit as st
import numpy as np
from seg2med_app.app_utils.image_utils import load_image_canonical
def combine_selected_organs(uploaded_file):
    if uploaded_file.name.endswith(".zip"):
        valid_organ_paths = st.session_state["valid_organ_paths"]
        multi_seg = np.zeros(st.session_state["volume_shape"], dtype=np.uint8)
        for organ_name in st.session_state["selected_organs"]:
            label_value = st.session_state["organ_to_label"][organ_name]
            mask = load_image_canonical(valid_organ_paths[organ_name]).get_fdata()
            multi_seg[mask > 0] = label_value
    elif uploaded_file.name.endswith((".nii.gz", ".nii")):
        multi_seg = np.zeros(st.session_state["volume_shape"], dtype=np.uint8)
        print('combine selected organs:', st.session_state["selected_organs"])
        for organ_name in st.session_state["selected_organs"]:
            label_value = st.session_state["organ_to_label"][organ_name]
            multi_seg[st.session_state["combined_seg"] == label_value] = label_value
    return multi_seg