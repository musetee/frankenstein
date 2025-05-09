import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import os

def load_nii_and_save_png(nii_file_path, output_png_path, dpi=300):
    # Load the .nii.gz file
    #nii_file_path = 'path/to/your/file.nii.gz'  # Update with your file path
    nii_data = nib.load(nii_file_path)

    # Extract the volume data as a NumPy array
    volume = nii_data.get_fdata()

    # print(volume.shape)
    # Select a slice (e.g., the middle slice along the third dimension)
    slice_data = volume
    slice_data = np.rot90(slice_data, k=-1)  # Rotate clockwise

    # Normalize the slice data for visualization (optional)
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    slice_data = slice_data.astype(np.uint8)

    # Save the slice as a PNG image
    # output_png_path = 'output_slice.png'  # Update with your desired output path
    # plt.imsave(output_png_path, slice_data, cmap='gray')
    
    plt.figure(figsize=(8, 8))  # Set figure size
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')  # Turn off axes
    plt.savefig(output_png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    print(f"Slice saved as {output_png_path}")

folder_list = [
    {"model": "pix2pix", "folder": r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\ct_compare_4_models_256\20241027_0459_Infer_pix2pix'},
    {"model": "cycle_gan", "folder": r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\ct_compare_4_models_256\20241028_1444_Infer_cycle_gan'},
    {"model": "unet", "folder": r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\ct_compare_4_models_256\20241028_1917_Infer_AttentionUnet'},
    {"model": "ddpm", "folder": r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\ct_compare_4_models_256\20241031_1129_Infer_ddpm2d_seg2med_noManualAorta'}
]

output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\output_png'
patient_ID_list = ['1PC082']
slice_ID_list = [30]
for item in folder_list:
    print(f"save sample for Model: {item['model']}, Folder: {item['folder']}")
    model_name = item['model']
    folder = item['folder']
    slice_folder = os.path.join(folder, 'saved_outputs', 'slice_output')
    for patient_ID in patient_ID_list:
        for slice_ID in slice_ID_list:
            synthsized_image_name = f"{patient_ID}_synthesized_{slice_ID}.nii.gz"
            synthsized_image_path = os.path.join(slice_folder, synthsized_image_name)
            synthsized_output_filename = f"{model_name}_{patient_ID}_synthsized_{slice_ID}.png"
            synthesized_output_path = os.path.join(output_folder, synthsized_output_filename)
            load_nii_and_save_png(synthsized_image_path, synthesized_output_path)

            target_image_name = f"{patient_ID}_target_{slice_ID}.nii.gz"
            target_image_path = os.path.join(slice_folder, target_image_name)
            target_output_filename = f"{model_name}_{patient_ID}_target_{slice_ID}.png"
            target_output_path = os.path.join(output_folder, target_output_filename)
            load_nii_and_save_png(target_image_path, target_output_path)

            seg_image_name = f"{patient_ID}_seg_{slice_ID}.nii.gz"
            seg_image_path = os.path.join(slice_folder, seg_image_name) 
            seg_output_filename = f"{model_name}_{patient_ID}_seg_{slice_ID}.png"
            seg_output_path = os.path.join(output_folder, seg_output_filename)    
            load_nii_and_save_png(seg_image_path, seg_output_path)