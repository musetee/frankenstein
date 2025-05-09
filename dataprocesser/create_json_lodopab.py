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

def create_metadata_jsonl_lodopab(base_path, mode='train', output_json_file= 'lodopab_dataset.json'):
    ground_truth_path = os.path.join(base_path, 'ground_truth_'+mode)
    observation_path = os.path.join(base_path, 'observation_'+mode)

    # Initialize dataset list
    dataset_list = []

    # Iterate through the ground truth files
    for gt_file in os.listdir(ground_truth_path):
        if is_image_file(gt_file):
            # Get the corresponding observation file
            obs_file = gt_file.replace('ground_truth', 'observation')
            
            # Create the entry
            entry = {
                'ground_truth': os.path.join(ground_truth_path, gt_file),
                'observation': os.path.join(observation_path, obs_file)
            }
            
            # Append to the dataset list
            dataset_list.append(entry)

    # Save the dataset list as a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(dataset_list, json_file, indent=4)

    print(f'Dataset list saved to lodopab_dataset.json with {len(dataset_list)} entries.')
   
def read_metadata_jsonl(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def print_json_info(data_info):
        for entry in tqdm(data_info, desc="Calculating slice info"):
            print(entry['patient_name'])
            
if __name__ == '__main__':
    base_path = r"F:\yang_Projects\Datasets\LoDoPaB"
    create_metadata_jsonl_lodopab(base_path, mode='train', output_json_file= './data_table/lodopab_dataset.json')