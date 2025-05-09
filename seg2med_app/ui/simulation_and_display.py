# seg2med_app/ui/simulation_and_display.py
import os
import pandas as pd
import streamlit as st
from seg2med_app.simulation.simulator import simulate_modality
from seg2med_app.ui.mri_sequence_controls import add_custom_sequence
from seg2med_app.app_utils.titles import *

import hashlib
import datetime
def hash_current_time():
    now_str = datetime.datetime.now().isoformat()  # 形如 '2025-05-07T15:30:12.123456'
    return hashlib.md5(now_str.encode()).hexdigest()

def simulation_controls(app_root):
    if "combined_seg" not in st.session_state:
        st.warning("⚠️ Please random load or upload a segmentation first.")
        return

    combined_seg = st.session_state["combined_seg"]
    contour = st.session_state.get("contour", None)
    input_modality = st.session_state.get("input_modality", "ct")

    add_custom_sequence()

    col5, col6, col7 = st.columns([2, 1, 2])

    with col5:
        selected_organs = st.multiselect("Select organs to view", options=st.session_state.get("organ_names", []))
        st.session_state["selected_organs"] = selected_organs
    with col6:
        modality_options = ["CT", "T1_GRE", "T2_SPACE", "T1_VIBE_IN", "T1_VIBE_OPP", "T1_VIBE_DIXON"]
        custom_options = list(st.session_state.get("custom_sequences", {}).keys())
        all_modality_options = modality_options + custom_options
        modality = st.selectbox("Output modality", all_modality_options)
        print('select output modality:', modality)
        
        if modality in custom_options:
            custom_signal_fn = st.session_state["custom_sequences"][modality]["fn"]
            print("test custom fn,", custom_signal_fn(T1=1000, T2=100, rho=1.))
            modality_idx = 10000
        else:
            custom_signal_fn = None
            modality_idx = modality_options.index(modality)
        
        print('input modality:', input_modality)
        
        from seg2med_app.ui.model_card import model_options
        selected_model = st.selectbox("Checkpoint", model_options, index=0)
        st.session_state["selected_model"] = selected_model
        
    with col7:
        param_file = st.file_uploader("Upload simulation CSV", type="csv")
        params_csv_ct = os.path.join(app_root, "simulation/params_ct.csv")
        params_csv_mr = os.path.join(app_root, "simulation/params_mr.csv")
        if param_file:
            df_params = pd.read_csv(param_file)
        else:
            df_params = pd.read_csv(params_csv_ct if modality == "CT" else params_csv_mr)
        st.dataframe(df_params.head(), use_container_width=True)  # 只显示前5行

    st.session_state["modality_idx"] = modality_idx

    if "existed_simulation_hash" not in st.session_state:
        st.session_state["existed_simulation_hash"] = ""
        st.session_state["current_simulation_hash"] = ""
        
    make_step_renderer(step6_frankenstein)
    if st.button("⚙️ Run Simulation"):
        st.info("Running simulation...")
        print("Running simulation...")
        st.session_state["current_simulation_hash"] = hash_current_time()
        
    # only run simulation after clicking button
    if st.session_state["current_simulation_hash"] != st.session_state["existed_simulation_hash"]: 
        st.session_state["existed_simulation_hash"] = st.session_state["current_simulation_hash"]
        output = simulate_modality(contour, modality_idx, combined_seg, input_modality, params_csv_ct, params_csv_mr, custom_signal_fn=custom_signal_fn)
        st.session_state["processed_img"] = output
        st.success("Simulation finished ✅")
        
    