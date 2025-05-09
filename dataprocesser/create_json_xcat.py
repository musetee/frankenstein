from tqdm import tqdm
import json

from configs import config as cfg
import os

VERBOSE = cfg.verbose
from collections import defaultdict

IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz',
    '.hdf5',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def create_metadata_jsonl_xcat(base_path, 
                               mode='train', 
                               sino_entry = "_sino_Metal.nrrd",
                               img_entry = "_img_GT_noNoise.nrrd", 
                               output_json_file= 'xcat_dataset.json'):
    
    train_set_path = os.path.join(base_path, mode)
    # Initialize dataset list
    dataset_list = []
    prefixes = set()
    for filename in os.listdir(train_set_path):
        if is_image_file(filename):
            prefix = filename.split('_')[0]
            prefixes.add(prefix)
    prefixes = sorted(prefixes)
    
    for prefix in prefixes:
        sino_path = os.path.join(train_set_path, prefix + sino_entry)
        img_path = os.path.join(train_set_path, prefix + img_entry)

        # Create the entry
        entry = {
            'ground_truth': img_path,
            'observation': sino_path
        }
        
        # Append to the dataset list
        dataset_list.append(entry)

    # Save the dataset list as a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(dataset_list, json_file, indent=4)

    print(f'Dataset list saved to xcat_dataset.json with {len(dataset_list)} entries.')
   
def read_metadata_jsonl(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def print_json_info(data_info):
        for entry in tqdm(data_info, desc="Calculating slice info"):
            print(entry['patient_name'])
            
if __name__ == '__main__':
    base_path = r"F:\yang_Projects\ICTUNET_torch\datasets"
    create_metadata_jsonl_xcat(base_path, 
                               mode='train', 
                               sino_entry = "_sino_Metal.nrrd",
                               img_entry = "_img_GT_noNoise.nrrd", 
                               output_json_file= './data_table/xcat_dataset.json')