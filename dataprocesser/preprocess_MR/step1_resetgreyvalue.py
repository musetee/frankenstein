import os
import numpy as np
import nrrd

def load_nrrd_file(nrrd_path):
    data, header = nrrd.read(nrrd_path)
    return data, header

def save_nrrd_file(data, header, save_path):
    nrrd.write(save_path, data, header)

def process_and_overlay(nrrd_path1, save_path):
    data1, header1 = load_nrrd_file(nrrd_path1)
    
    # 将 data1 中的灰度值 1, 2, 3 各加上 57，而背景（0）不变
    data1 = np.where(data1 == 1, data1 + 98, data1)
    data1 = np.where(data1 == 2, data1 + 197, data1)
    data1 = np.where(data1 == 3, data1 + 296, data1)

    save_nrrd_file(data1, header1, save_path)

def main(folder1, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = set(os.listdir(folder1))


    for filename in files1:
        if filename.endswith(".nrrd"):
            nrrd_path1 = os.path.join(folder1, filename)
            save_path = os.path.join(output_folder, filename)

            print(f"Processing {nrrd_path1}, saving to {save_path}")
            process_and_overlay(nrrd_path1, save_path)

if __name__ == "__main__":
    
    folder1 = 'MR_VIBE_seg2_nrrd_24_resetregion'  
    output_folder = 'MR_VIBE_seg2_nrrd_24_resetregion&greyvalue'  

    main(folder1, output_folder)