import os
import numpy as np
import nrrd
from image_function_save_png_hist import load_volume, save_volume, pad_image, pad_image_3D
import re
def load_nrrd_file(nrrd_path):
    data, header = nrrd.read(nrrd_path)
    return data, header

def save_nrrd_file(data, header, save_path):
    nrrd.write(save_path, data, header)


def extract_liver_region(nrrd_path1, save_path):
    data1, headeroraffine = load_volume(nrrd_path1)
    
    # 将灰度值为5的部分保留并设为1，其他部分设为0，假设5为肝脏区域的标签
    data1 = np.where(np.isclose(data1, 6, atol=1e-6), 1, 0).astype(np.uint8)
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
            if "target_volume" in filename:
                nrrd_path1 = os.path.join(folder1, filename)

                patient_id = 'synthetic'
                output_file_name = f"{patient_id}_liver_mask_volume.nii.gz"
                final_save_path = os.path.join(output_folder, output_file_name)

                print(f"Processing {nrrd_path1}, saving to {final_save_path}")
                extract_liver_region(nrrd_path1, final_save_path)

if __name__ == "__main__":
    folder1 = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct\saved_outputs\volume_output'  
    output_folder = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct\saved_outputs\volume_output'

    main(folder1, output_folder)