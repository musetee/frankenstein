import matplotlib
matplotlib.use('TKAgg')
import os
from image_function_save_png_hist import load_nii_and_save_gt_synth_percentage_hist, load_nii_and_save_png, calculate_histcc
synthesized_name = 'synthrad'
if synthesized_name == 'mr':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_mr' 
    folder_list = [
        {"model": "ddpm", "folder": '20241121_1055_Infer_ddpm2d_seg2med_mr_512'},
    ]
    pID_slice_list = [
        {"patient": "1PC085", "slice": 74, "FOV": [256,256,120]},
        #{"patient": "1PC084", "slice": 52, "FOV": [300,220,120]},
        #{"patient": "1PC096", "slice": 43, "FOV": [140,240,120]},
        #{"patient": "1PC096", "slice": 19, "FOV": [140,240,120]},
        #{"patient": "1PC095", "slice": 37, "FOV": [140,240,120]},
        #{"patient": "1PC084", "slice": 76, "FOV": [256,256,120]},
        #{"patient": "1PC095", "slice": 37, "FOV": [300,256,120]},
        #{"patient": "1PC082", "slice": 50, "FOV": [300,256,120]},
    ]
elif synthesized_name == 'synthrad':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models'
    folder_list = [
        {"model": "ddpm", "folder": '20241119_0052_Infer_ddpm2d_seg2med_synthrad_512'},
        #{"model": "cycle_gan", "folder": '20241128_1141_Infer_cycle_gan_synthrad_512'},
        #{"model": "pix2pix", "folder": '20241125_2055_Infer_pix2pix_synthrad_512'},
        #{"model": "unet", "folder": '20241125_2116_Infer_AttentionUnet_synthrad_512'},
    ]
    pID_slice_list = [
        #{"patient": "1PC098", "slice": 91, "FOV": [256,256,120]},
        #{"patient": "1PC084", "slice": 52, "FOV": [300,220,120]},
        #{"patient": "1PC096", "slice": 45, "FOV": [140,240,120]},
        #{"patient": "1PC096", "slice": 44, "FOV": [140,240,120]},
       # {"patient": "1PC084", "slice": 76, "FOV": [256,256,120]},
        #{"patient": "1PC095", "slice": 37, "FOV": [300,256,120]},
        #{"patient": "1PC098", "slice": 27, "FOV": [140,240,120]},
        {"patient": "1PC088", "slice": 52, "FOV": [256,256,120]},
        #{"patient": "1PC085", "slice": 86, "FOV": [256,256,120]},
    ]
elif synthesized_name == 'synthrad256':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models'
    folder_list = [
        {"model": "ddpm", "folder": '20241031_1129_Infer_ddpm2d_seg2med_noManualAorta_256'},
        {"model": "cycle_gan", "folder": '20241028_1444_Infer_cycle_gan_256'},
        {"model": "pix2pix", "folder": '20241027_0459_Infer_pix2pix_256'},
        {"model": "unet", "folder": '20241028_1917_Infer_AttentionUnet_256'},
    ]
    pID_slice_list = [
        #{"patient": "1PC098", "slice": 91, "FOV": [256,256,120]},
        #{"patient": "1PC084", "slice": 52, "FOV": [300,220,120]},
        #{"patient": "1PC096", "slice": 43, "FOV": [140,240,120]},
        {"patient": "1PC096", "slice": 44, "FOV": [70,120,60]},
        #{"patient": "1PC095", "slice": 37, "FOV": [140,240,120]},
    ]
elif synthesized_name == 'anish':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models'
    folder_list = [
        {"model": "ddpm", "folder": '20241119_1142_Infer_ddpm2d_seg2med_anish_512'},
        {"model": "cyclegan", "folder": '20241128_1159_Infer_cycle_gan_anish_512'},
        {"model": "pix2pix", "folder": '20241128_1342_Infer_pix2pix_anish_512'},
        {"model": "unet", "folder": '20241128_1354_Infer_AttentionUnet_anish_512'},

    ]
    pID_slice_list = [
        #{"patient": "380", "slice": 192, "FOV": [256,256,120]},
        #{"patient": "v232", "slice": 158, "FOV": [256,256,120]},
        #{"patient": "v242", "slice": 130, "FOV": [256,256,120]},
        {"patient": "380", "slice": 192, "FOV": [256,256,120]},
    ]
elif synthesized_name == 'anish256':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models'
    folder_list = [
        {"model": "ddpm", "folder": '20241031_1129_Infer_ddpm2d_seg2med_noManualAorta_256'},
        {"model": "cycle_gan", "folder": '20241028_1444_Infer_cycle_gan_256'},
        {"model": "pix2pix", "folder": '20241027_0459_Infer_pix2pix_256'},
        {"model": "unet", "folder": '20241028_1917_Infer_AttentionUnet_256'},
    ]
    pID_slice_list = [
        {"patient": "380", "slice": 173, "FOV": [128,128,60]},
        #{"patient": "v232", "slice": 158, "FOV": [128,128,60]},
        #{"patient": "v242", "slice": 130, "FOV": [128,128,60]},
    ]
elif synthesized_name == 'ct2mr2ct':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_ctmr'
    folder_list = [
        {"model": "ddpm_ct2mr", "folder": '20241226_1148_Infer_ddpm2d_seg2med_ct2mr_512'},
        #{"model": "ddpm_mr2ct", "folder": '20241119_2030_Infer_ddpm2d_seg2med_mr2ct_512'},
    ]
    pID_slice_list = [
        #{"patient": "1PC096", "slice": 44, "FOV": [140,240,120]},
        #{"patient": "1PC084", "slice": 76, "FOV": [256,256,120]},
        #{"patient": "1PC088", "slice": 51, "FOV": [256,256,120]},
        {"patient": "1PC095", "slice": 37, "FOV": [300,256,120]},
        {"patient": "1PC084", "slice": 76, "FOV": [256,256,120]},
    ]

elif synthesized_name == 'ct2mr2ct256':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase3_ctmr'
    folder_list = [
        #{"model": "ddpm_ct2mr", "folder": '20241117_2249_Infer_ddpm2d_seg2med_ct2mr_256'},
        {"model": "ddpm_mr2ct", "folder": '20241117_2326_Infer_ddpm2d_seg2med_mr2ct_256'},
    ]
    pID_slice_list = [
        #{"patient": "1PC096", "slice": 44, "FOV": [140,240,120]},
        {"patient": "1PC084", "slice": 76, "FOV": [256,256,120]},
        #{"patient": "1PC088", "slice": 51, "FOV": [256,256,120]},
        {"patient": "1PC095", "slice": 37, "FOV": [300,256,120]},
        {"patient": "1PC082", "slice": 50, "FOV": [300,256,120]},
    ]
output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\output_png'

only_save_fov = False
save_hist = False
fov_box = True
pad_size = [512, 512]
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
        
        
        load_nii_and_save_png(target_image_path, target_output_path, gt_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size, pad_size=pad_size)
        load_nii_and_save_png(seg_image_path, seg_output_path, seg_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size, pad_size=pad_size)
        load_nii_and_save_png(synthsized_image_path, synthesized_output_path, output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size, pad_size=pad_size)

        output_hist_filename = f"{model_name}_{patient_ID}_{slice_ID}_hist.png"
        output_hist_path = os.path.join(output_folder,output_hist_filename)

        if save_hist:
            if synthesized_name == 'ct2mr2ct' or synthesized_name == 'mr':
                range = [1,255] # [-500,1000]
                bins = 128
            else:
                range = [-500,1000]
                bins = 256
            load_nii_and_save_gt_synth_percentage_hist(target_image_path, synthsized_image_path, output_hist_path, slice_index=None, bins=bins, range=range) 
            histcc = calculate_histcc(target_image_path, synthsized_image_path, slice_index=None, bins=bins, range=range) # 
            print("Histogram Correlation Coefficient (HistCC):", histcc)

        #image_fov_from_volume_slice(nii_file_path, output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(gt_nii_file_path, gt_output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(seg_nii_file_path, seg_output_png_path, slice_ID, center_x, center_y, fov_size)