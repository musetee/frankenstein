import os
import numpy as np
import nrrd

def load_nrrd_file(nrrd_path):
    data, header = nrrd.read(nrrd_path)
    return data, header

def save_nrrd_file(data, header, save_path):
    nrrd.write(save_path, data, header)

def overlay_images(mask_data, organ_data):
    # Combine the images by adding the pixel values
    combined_data = mask_data + organ_data
    return combined_data

def process_and_overlay(nrrd_path1, nrrd_path2, save_path):
    data1, header1 = load_nrrd_file(nrrd_path1)
    data2, header2 = load_nrrd_file(nrrd_path2)
    
    combined_data = overlay_images(data1, data2)
    save_nrrd_file(combined_data, header1, save_path)

def main(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    common_files = files1.intersection(files2)

    for filename in common_files:
        if filename.endswith(".nrrd"):
            nrrd_path1 = os.path.join(folder1, filename)
            nrrd_path2 = os.path.join(folder2, filename)
            save_path = os.path.join(output_folder, filename)

            print(f"Processing {nrrd_path1} and {nrrd_path2}, saving to {save_path}")
            process_and_overlay(nrrd_path1, nrrd_path2, save_path)

if __name__ == "__main__":
    
    folder1 = 'MR_VIBE_contour_nrrd'  
    folder2 = 'MR_VIBE_seg_nrrd'  
    output_folder = 'MR_VIBE_contour_seg_nrrd'  

    main(folder1, folder2, output_folder)