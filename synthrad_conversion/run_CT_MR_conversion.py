# total steps for seg2med
import sys
import argparse
import os
from dataprocesser import step1_init_data_list 
# Add project directories to the system path

def run(input_args=None, mode='ct2mr'):
    steps_conducted = {
        #'step1_convert_mask',
        'TrainOrTest'

    }
    if 'step1_convert_mask' in steps_conducted:
        from dataprocesser import Preprocess_MRCT_mask_conversion as maskconvert
        synthrad_root = r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis'
        patient_list = ['1PC082', '1PC084', '1PC085', '1PC088', '1PC092', '1PC093', '1PC095', '1PC096', '1PC097', '1PC098']
        csvfile = f'{mode}_conversion_ckm4gpu.csv'
        maskconvert.run_mask_conversion_synthrad_test(synthrad_root, patient_list, mode, csvfile)
    
    if 'TrainOrTest' in steps_conducted:
        from synthrad_conversion import train
        dataset_name = 'mr2ct_simplified_csv' if mode=='mr2ct' else 'combined_simplified_csv_seg_mr_loader'
        train.run(input_args, dataset_name)



if __name__ == "__main__":
    run()