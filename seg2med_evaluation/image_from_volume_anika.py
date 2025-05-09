import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import os
import nrrd
from image_fov_from_volume_slice import image_fov_from_volume_slice
from image_function_save_png_hist import load_nii_and_save_gt_synth_percentage_hist, load_nii_and_save_png, calculate_histcc
synthesized_name = 'anika'
if synthesized_name == 'anika':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models'
    folder_list = [
            {"model": "ddpm", "folder": 'Infer_ddpm2d_seg2med_anika_512_all'},
            {"model": "cycle_gan", "folder": '20241128_1337_Infer_cycle_gan_anika_512'},
            {"model": "pix2pix", "folder": '20241128_1340_Infer_pix2pix_anika_512'},
            {"model": "unet", "folder": '20241128_1347_Infer_AttentionUnet_anika_512'}
        ]
    pID_slice_list = [
        {"patient": "34438427_5", "slice": 48, "FOV": [256,300,160]},
        #{"patient": "34438427_5", "slice": 50, "FOV": [256,300,160]},
        #{"patient": "40293225_16", "slice": 40, "FOV": [256,300,160]},
    ]
elif synthesized_name == 'anika256':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models'
    folder_list = [
            {"model": "ddpm", "folder": '20241103_0437_Infer_ddpm2d_seg2med_Anika_CT'},
            {"model": "cycle_gan", "folder": '20241103_1650_Infer_cycle_gan_Anika_CT'},
            {"model": "pix2pix", "folder": '20241103_0422_Infer_pix2pix_Anika_CT'},
            {"model": "unet", "folder": '20241103_1627_Infer_AttentionUnet_Anika_CT'}
        ]
    pID_slice_list = [
        {"patient": "34438427_5", "slice": 48, "FOV": [128,150,60]},
        {"patient": "40094015_4", "slice": 44, "FOV": [128,128,60]},
    ]
output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\output_png'

save_hist = False
only_save_fov = False
for item in folder_list:
    print(f"save sample for Model: {item['model']}, Folder: {item['folder']}")
    model_name = item['model']
    folder = item['folder']
    slice_folder = os.path.join(root, folder, 'saved_outputs', 'volume_output')
    
    for slice_item in pID_slice_list:
        patient_ID = slice_item['patient']
        slice_ID = slice_item['slice']
        FOV_info = slice_item['FOV']
        center_x = FOV_info[0]
        center_y = FOV_info[1]
        fov_size = FOV_info[2]

        target_image_name = f"{patient_ID}_target_volume.nii.gz"
        target_image_path = os.path.join(slice_folder, target_image_name)
        target_output_filename = f"{model_name}_{patient_ID}_{slice_ID}_target_volume.png"
        target_output_path = os.path.join(output_folder, target_output_filename)
        
        seg_image_name = f"{patient_ID}_seg_volume.nii.gz"
        seg_image_path = os.path.join(slice_folder, seg_image_name) 
        seg_output_filename = f"{model_name}_{patient_ID}_{slice_ID}_seg_volume.png"
        seg_output_path = os.path.join(output_folder, seg_output_filename)    
        
        synthsized_image_name = f"{patient_ID}_synthesized_volume.nii.gz"
        synthsized_image_path = os.path.join(slice_folder, synthsized_image_name)
        synthsized_output_filename = f"{model_name}_{patient_ID}_{slice_ID}_synthsized_volume.png"
        synthesized_output_path = os.path.join(output_folder, synthsized_output_filename)
        

        nii_file_path = synthsized_image_path
        output_png_filename = f"{model_name}_{patient_ID}_{slice_ID}_fov_synthsized_volume.png"
        output_fov_png_path = os.path.join(output_folder, output_png_filename)
        
        gt_nii_file_path = target_image_path
        gt_output_png_filename = f"{model_name}_{patient_ID}_{slice_ID}_fov_target_volume.png"
        gt_output_fov_png_path = os.path.join(output_folder, gt_output_png_filename)

        seg_nii_file_path = seg_image_path
        seg_output_png_filename = f"{model_name}_{patient_ID}_{slice_ID}_fov_seg_volume.png"
        seg_output_fov_png_path = os.path.join(output_folder, seg_output_png_filename)
        fov_box=True
        load_nii_and_save_png(target_image_path, target_output_path, gt_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)
        load_nii_and_save_png(seg_image_path, seg_output_path, seg_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)
        load_nii_and_save_png(synthsized_image_path, synthesized_output_path, output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)

        output_hist_filename = f"{model_name}_{patient_ID}_{slice_ID}_hist.png"
        output_hist_path = os.path.join(output_folder,output_hist_filename)

        if save_hist:
            load_nii_and_save_gt_synth_percentage_hist(target_image_path, synthsized_image_path, output_hist_path, slice_index=None, range=[-500,1000])
            histcc = calculate_histcc(target_image_path, synthsized_image_path, slice_index=None, range=[-500,1000])
            print("Histogram Correlation Coefficient (HistCC):", histcc)

        #image_fov_from_volume_slice(nii_file_path, output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(gt_nii_file_path, gt_output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(seg_nii_file_path, seg_output_png_path, slice_ID, center_x, center_y, fov_size)