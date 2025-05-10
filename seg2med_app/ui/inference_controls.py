from synthrad_conversion.networks.ddpm.ddpm_mri2ct import DiffusionModel_multimodal_for_app
from synthrad_conversion.utils.my_configs_yacs import init_cfg, config_path
from synthrad_conversion.train import is_linux, setup
from seg2med_app.app_utils.image_utils import processing_slice_to_right_orientation
import streamlit as st
import numpy as np
import torch
import os 


    
def inference_controls():
    if st.button("⚙️ Run inference"):
        st.info("Running inference...")
        opt =init_cfg('tutorial8_app_inference.yaml') 
        
        islinux = is_linux()
        if  islinux and torch.cuda.device_count() > 1:
            print("🟢 Detected Linux with multiple GPUs — using DDP...")
            world_size = torch.cuda.device_count()
            opt.is_ddp = True
            opt.rank = int(os.environ["LOCAL_RANK"])
            opt.world_size = world_size
            setup(opt.rank, world_size)

        else:
            print("🟡 Using single-GPU training (Windows or single GPU)...")
            opt.is_ddp = False
            opt.rank = 0
            opt.world_size = 1
        
        model_name_path = 'app'
        paths=config_path(model_name_path)
        opt.ckpt_path = os.path.join('seg2med_app','model', f'{st.session_state["selected_model"]}_model_100.pt')
        print('selected ckpt:', opt.ckpt_path)
        model = DiffusionModel_multimodal_for_app(
            opt, paths,
            None, None, None, None
        )
        source = st.session_state["processed_img"][:, :, st.session_state["z_idx"]]
        #
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(source, cmap="gray")
        plt.title("Does this look anatomically correct?")
        plt.xlabel("Left → Right?")
        plt.ylabel("Back → Front?")
        plt.grid(False)
        plt.savefig(r'seg2med_app\checkinput.png')
        plt.close()
        modality = st.session_state["modality_idx"]
        print("modality into model: ", modality)
        if modality in range(6): # accepted range by model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def streamlit_callback(current_step, total_steps):
                percent = current_step / total_steps
                progress_bar.progress(percent)
                status_text.text(f"Inference Step {current_step}/{total_steps}")
                
            output_batch = model.inference(source, modality, streamlit_callback)
            output_img = output_batch.cpu().detach().numpy().squeeze().squeeze()
            print("output image shape", output_img.shape)
            st.session_state["output_img"] = processing_slice_to_right_orientation(output_img)
            st.write("inference result")
            
        else:
            print("modality not accepted!")

        st.session_state["processed_img"]
        st.success("inference finished ✅")
        
        