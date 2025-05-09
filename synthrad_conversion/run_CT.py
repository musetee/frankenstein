# total steps for seg2med
import sys
import argparse
import os
from dataprocesser import step1_init_data_list 
# Add project directories to the system path

def run(input_args=None):
    steps_conducted = {
        #'step1_create_data_list',
        'step2_create_seg',
        #'step_optional_replace_Anish_csv',
        #'step3_create_csv',
        #'step4_testdata',
        #'step5_savemask',
        #'step_optional_slicing'
        #'TrainOrTest'

    }
    if 'step1_create_data_list' in steps_conducted:
        from dataprocesser import step2_create_segmentation as createseg
        createseg.create_dataset_list()

    if 'step2_create_seg' in steps_conducted:
        from dataprocesser import step2_create_segmentation as createseg
        import torch
        gpu_index = 1
        device = 'cuda'
        task = 'tissue_types' # 'total'
        dataset = 'anish' # anika_all_ct anish synthrad_ct
        GPU_ID=[gpu_index]
        
        # cuda:{GPU_ID[0]}
        os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpu_index}'
        device = 'cuda' #torch.device(f'cuda' if torch.cuda.is_available() else 'cpu') # 0=TitanXP, 1=P5000
        
        print('use GPU: ', torch.cuda.get_device_name(f'cuda'))
        createseg.run(dataset=dataset, device=device, task=task) 

    if 'step_optional_replace_Anish_csv' in steps_conducted:
        input_csv=r'synthrad_conversion\healthy_dissec_helix.csv'
        output_csv=r'synthrad_conversion\combined_new_helix.csv'
        step1_init_data_list.list_replace_anish_csv(input_csv, output_csv)

    if 'step3_create_csv' in steps_conducted:
        mode='createAnikaCSV'
        if mode =='combineSynthradAnishCSV':
            data1=r'/gpfs/bwfor/work/ws/hd_qf295-foo/synthrad/Task1/pelvis'
            data2=r'synthrad_conversion/healthy_dissec_helix.csv'
            #data1=r'D:\Projects\data\synthrad\train\Task1\pelvis'
            #data2=r'synthrad_conversion/healthy_dissec_home.csv'
            combined_csv=r'synthrad_conversion/combined_new_home.csv'
            step1_init_data_list.create_csv_combine_lists_synthrad_anish(data1, data2, combined_csv)

        elif mode == 'createAnikaCSV':
            ct_dir = r'E:\Projects\yang_proj\data\anika\CT'
            mri_dir = r'E:\Projects\yang_proj\data\anika\MR_registrated'
            ct_csv=r'synthrad_conversion/combined_anika_ct_newserver.csv'
            mr_csv=r'synthrad_conversion/combined_anika_mr_newserver.csv'
            step1_init_data_list.create_csv_Anika(ct_dir, mri_dir, ct_csv, mr_csv)
            
    if 'step_optional_slicing' in steps_conducted: # optinal step of slicing 
        from dataprocesser import step4_save_intermediate_dataset as slicing
        slicing.run()

    if 'step4_testdata' in steps_conducted: # step test data
        from dataprocesser import step6_data_test as data_test
        data_test.run(dataset = 'combined_simplified_csv_seg_assigned')

    if 'step5_savemask' in steps_conducted: # step for evaluation
        from dataprocesser import step7_save_mask as save_mask
        save_mask.run(dataset = 'combined_simplified_csv_seg_assigned')
    
    if 'TrainOrTest' in steps_conducted:
        from synthrad_conversion import train
        train.run(input_args, dataset_name = 'combined_simplified_csv_seg_assigned')

if __name__ == "__main__":
    run()