import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import os
import nrrd
from utils import pad_image

def load_nii_and_save_png(nii_file_path, output_png_path, output_fov_png_path, slice_index, dpi=300, 
                          fov_box=False, center_x=0, center_y=0, fov_size=60,
                          ):
    # Load the .nii.gz file
    #nii_file_path = 'path/to/your/file.nii.gz'  # Update with your file path
    if nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz'):
        nii_data = nib.load(nii_file_path)
        volume = nii_data.get_fdata()
    elif nii_file_path.endswith('.nrrd'):
    # Load the NRRD files
        volume, _ = nrrd.read(nii_file_path)

    # print(volume.shape)
    # Select a slice (e.g., the middle slice along the third dimension)
    slice_data = volume[:, :, slice_index]
    slice_data = np.rot90(slice_data, k=-1)  # Rotate clockwise
    slice_data = np.fliplr(slice_data)
    slice_data = pad_image(slice_data, desired_size=[512,512],pad_value=-1000)
    # Normalize the slice data for visualization (optional)
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    slice_data = slice_data.astype(np.uint8)

    # Save the slice as a PNG image
    # output_png_path = 'output_slice.png'  # Update with your desired output path
    # plt.imsave(output_png_path, slice_data, cmap='gray')
    # Create a rectangle patch
    
    fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size
    ax.imshow(slice_data, cmap='gray', vmin=0, vmax=255)

    if fov_box:
        x_start = max(0, center_x - fov_size // 2)
        y_start = max(0, center_y - fov_size // 2)
        rect = plt.Rectangle((x_start, y_start), fov_size, fov_size, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)  # Add the rectangle to the plot

        # Calculate the boundaries of the square FOV
        x_start = max(0, center_x - fov_size // 2)
        x_end = min(slice_data.shape[1], center_x + fov_size // 2)
        y_start = max(0, center_y - fov_size // 2)
        y_end = min(slice_data.shape[0], center_y + fov_size // 2)

        # Crop the square FOV
        cropped_fov = slice_data[y_start:y_end, x_start:x_end]

        # Save the cropped FOV as a PNG image
        plt.figure(figsize=(8, 8))  # Set figure size (adjust as needed)
        plt.imshow(cropped_fov, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # Turn off axes
        plt.savefig(output_fov_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

        print(f"Cropped FOV saved as {output_fov_png_path}")

    plt.axis('off')  # Turn off axes
    plt.savefig(output_png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    print(f"Slice saved as {output_png_path}")
synthesized_name = 'synthetic'
if synthesized_name == 'xact':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic'
    folder_list = [
        {"model": "ddpm", "folder": '20241119_0028_Infer_ddpm2d_seg2med_XCAT_CT_56Models_64slices_512'}
    ]
    pID_slice_list = [
        #{"patient": "Model76_Energy90", "slice": 35, "FOV": [256,220,120]},
        #{"patient": "Model89_Energy90", "slice": 18, "FOV": [256,220,120]},
        {"patient": "Model106_Energy90", "slice": 26, "FOV": [256,220,120]},
        #{"patient": "Model106_Energy90", "slice": 18, "FOV": [256,220,120]},
        #{"patient": "Model140_Energy90", "slice": 33, "FOV": [256,220,120]},
    ]
elif synthesized_name == 'synthetic':
    root = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic'
    folder_list = [
        {"model": "ddpm", "folder": '20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct'}
    ]
    pID_slice_list = [
        #{"patient": "synthetic", "slice": 14, "FOV": [256,220,120]},
        #{"patient": "synthetic", "slice": 40, "FOV": [256,220,120]},
        #{"patient": "synthetic", "slice": 2, "FOV": [256,220,120]},
        {"patient": "synthetic", "slice": 7, "FOV": [256,220,120]},
    ]
output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\output_png'

only_save_fov = False
fov_box = False
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
        
        load_nii_and_save_png(target_image_path, target_output_path, gt_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)
        load_nii_and_save_png(seg_image_path, seg_output_path, seg_output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)
        load_nii_and_save_png(synthsized_image_path, synthesized_output_path, output_fov_png_path, slice_ID, fov_box=fov_box, center_x=center_x, center_y=center_y, fov_size=fov_size)

    

        #image_fov_from_volume_slice(nii_file_path, output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(gt_nii_file_path, gt_output_png_path, slice_ID, center_x, center_y, fov_size)
        #image_fov_from_volume_slice(seg_nii_file_path, seg_output_png_path, slice_ID, center_x, center_y, fov_size)