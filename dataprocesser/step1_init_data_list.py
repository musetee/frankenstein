import os
import pandas as pd
import csv
from dataprocesser.dataset_registry import DATASET_REGISTRY
from synthrad_conversion.utils.my_configs_yacs import init_cfg, config_path

def init_dataset(opt=None, model_name_path=None, dataset = 'combined_simplified_csv_seg_assigned'):
    
    import os
    import shutil

    model_name_path = 'test' if model_name_path is None else model_name_path

    my_paths=config_path(model_name_path)
    create_folder = True
    if create_folder:
        os.makedirs(my_paths["saved_logs_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_model_folder"], exist_ok=True)
        os.makedirs(my_paths["tensorboard_log_dir"], exist_ok=True)
        os.makedirs(my_paths["saved_img_folder"], exist_ok=True)
        os.makedirs(my_paths["saved_inference_folder"], exist_ok=True)

    if opt is None:
        opt=init_cfg()
        opt.dataset.batch_size=16
        opt.dataset.val_batch_size=16
        opt.dataset.normalize='nonorm'
        opt.dataset.zoom=(0.5,0.5,1.0)
        opt.dataset.resized_size=(256,256, None)
        opt.dataset.div_size=(None,None,None)
        opt.dataset.WINDOW_WIDTH=2000
        opt.dataset.WINDOW_LEVEL=0
        opt.dataset.rotate=False
        #opt.dataset.windowing_and_shifting=True
        #opt.dataset.load_masks=True
        #opt.dataset.input_is_mask=False

    if dataset not in DATASET_REGISTRY:
        raise NotImplementedError(f"Dataset loader '{dataset}' not found.")
    
    monailoader = DATASET_REGISTRY[dataset](opt, my_paths)

    return monailoader, opt, my_paths

# Function to extract Patient ID from filename
def extract_patient_id(filename):
    import re
    # Regular expression to find the sequence of digits that represent the Patient ID
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0)
    return None


# Define the directory where your files are located
def appart_img_and_seg(all_files_list):

    # Use glob to get all files in the directory
    #all_files = glob.glob(f"{file_directory}/*.nii.gz")

    # Separate segmented and non-segmented files
    segmented_files = [file for file in all_files_list if 'seg' in file]
    orig_files = [file for file in all_files_list if 'seg' not in file]

    # Print the lists to verify
    print("Segmented files:")
    print(len(segmented_files))
    print("\nOrig files:")
    print(len(orig_files))

    return orig_files, segmented_files

def appart_seg_and_tissueseg(all_segs):
    seg_files = [file for file in all_segs if 'seg_tissue' not in file]
    tissue_seg_files = [file for file in all_segs if 'seg_tissue' in file and 'seg_seg_tissue' not in file]
    return seg_files, tissue_seg_files

def appart_merged_seg(all_segs):
    merged_seg_files = [file for file in all_segs if 'merged_seg' in file]
    return merged_seg_files
