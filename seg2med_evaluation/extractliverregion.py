import os
import numpy as np
import nrrd
from image_function_save_png_hist import load_volume, save_volume, pad_image, pad_image_3D

def load_nrrd_file(nrrd_path):
    data, header = nrrd.read(nrrd_path)
    return data, header

def save_nrrd_file(data, header, save_path):
    nrrd.write(save_path, data, header)


def extract_liver_region(nrrd_path1, save_path):
    data1, headeroraffine = load_volume(nrrd_path1)
    
    # 将灰度值为5的部分保留并设为1，其他部分设为0，假设5为肝脏区域的标签
    data1 = np.where(np.isclose(data1, 5, atol=1e-6), 1, 0).astype(np.uint8)
    #print(data1.shape)
    data1 = pad_image_3D(data1, [512,512, None], 0)
    #print(data1.shape)
    save_volume(data1, save_path, output_format='nii.gz', affine=headeroraffine)

def main(folder1, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = set(os.listdir(folder1))

    for filename in files1:
        if filename.endswith(".nrrd") or filename.endswith(".nii.gz") or filename.endswith(".nii") :
            nrrd_path1 = os.path.join(folder1, filename)
            # Substring to remove (normalize spaces)
            substring_to_remove = "i-Spiral 1.5 B30f - "
            
            # Normalize spaces in both strings
            file_name_normalized = " ".join(filename.split())  # Removes extra spaces
            substring_to_remove_normalized = " ".join(substring_to_remove.split())  # Removes extra spaces

            final_save_path = os.path.join(output_folder, file_name_normalized.replace('seg.nii', 'liver_mask_volume.nii').replace("ct-volume-", "").replace(substring_to_remove_normalized,""))

            print(f"Processing {nrrd_path1}, saving to {final_save_path}")
            extract_liver_region(nrrd_path1, final_save_path)

if __name__ == "__main__":
    folder1 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models\Infer_ddpm2d_seg2med_anika_512_all\saved_outputs\volume_output\seg'  
    output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models\Infer_ddpm2d_seg2med_anika_512_all\saved_outputs\volume_output\Liver_masks'

    main(folder1, output_folder)