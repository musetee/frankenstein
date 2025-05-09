# total steps for seg2med
import sys
import argparse
import os
from dataprocesser import step1_init_data_list 
# Add project directories to the system path
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  

def run(input_args=None):
    steps_conducted = {
        #'step1_create_data_list',
        #'step2_create_seg',
        #'step3_merge_segs'
        #'step_optional_replace_Anish_csv',
        #'step4_create_csv',
        #'step_optional_view_distribution',
        #'step5_preprocess'
        #'step6_testdata',
        #'step7_savemask',
        #'step_optional_slicing'
        'TrainOrTest'
        #'step_optional_take_preprocess_example'

    }
    
    mode='Synthrad'
    if 'step1_create_data_list' in steps_conducted or 'step2_create_seg' in steps_conducted:
        print('create_dataset_list')
        from dataprocesser import step2_create_segmentation as createseg
        synthrad_mr_file_list = createseg.create_dataset_list(dataset='synthrad_mr')
        anika_all_mr_file_list = createseg.create_dataset_list(dataset='anika_all_mr')
        dominik_all_mr_file_list = createseg.create_dataset_list(dataset='dominik')
        if mode == 'Synthrad_Anika':
            target_file_list = synthrad_mr_file_list + anika_all_mr_file_list
        elif mode == 'Synthrad':
            target_file_list = synthrad_mr_file_list
        elif mode == 'Anika':
            target_file_list = anika_all_mr_file_list
        elif mode == 'Dominik':
            target_file_list = dominik_all_mr_file_list
        print('dataset length', len(target_file_list))
        #print('dataset 2 length', len(anika_all_mr_file_list))

    if 'step2_create_seg' in steps_conducted:
        from dataprocesser import step2_create_segmentation as createseg
        gpu_index = 2
        GPU_ID=[gpu_index]
        
        # cuda:{GPU_ID[0]}
        device = torch.device(f'cuda:{GPU_ID[0]}' if torch.cuda.is_available() else 'cpu') # 0=TitanXP, 1=P5000
        os.environ["CUDA_VISIBLE_DEVICES"]='2'
        print('use GPU: ', torch.cuda.get_device_name(f'cuda:{GPU_ID[0]}'))
        device = 'cuda'
        createseg.run(target_file_list, 'total_mr', device) # total_mr task, will create segmentation with postfix '_seg_tissue'
        createseg.run(target_file_list, 'tissue_types_mr', device) # tissue_types_mr task, will create segmentation with postfix '_seg'
        

    if 'step_optional_replace_Anish_csv' in steps_conducted:
        input_csv=r'synthrad_conversion\healthy_dissec_helix.csv'
        output_csv=r'synthrad_conversion\combined_new_helix.csv'
        step1_init_data_list.list_replace_anish_csv(input_csv, output_csv)

    if 'step3_merge_segs' in steps_conducted:
        # add the two parts of segmentations from two Totalsegmentator MRI-tasks together
        if mode == 'Synthrad_Anika' or mode == 'Synthrad':
            synthrad_dir=r'E:\Projects\yang_proj\data\synthrad\Task1_val\pelvis'
            synthrad_segs, synthrad_pIDs = step1_init_data_list.list_img_pID_from_synthrad_folder(synthrad_dir, ["mr_seg"], None)
            synthrad_tissue_segs, synthrad_pIDs = step1_init_data_list.list_img_pID_from_synthrad_folder(synthrad_dir, ["mr_seg_tissue"], None)
        
        elif mode == 'Synthrad_Anika' or mode == 'Anika':
            anika_dir_mr = r'E:\Projects\yang_proj\data\anika\MR_registrated'
            anika_mr_list = step1_init_data_list.all_list_single_modality_from_anika_dataset_include_duplicate(anika_dir_mr)
            mr_files, anika_mr_seg_files = step1_init_data_list.appart_img_and_seg(anika_mr_list)
            anika_segs, anika_tissue_segs = step1_init_data_list.appart_seg_and_tissueseg(anika_mr_seg_files)

        elif mode == 'Dominik':
            dominik_dir_mr = r'E:\Projects\yang_proj\data\Dominik_MR_VIBE'
            dominik_all_mr_file_list = createseg.create_dataset_list(dataset='dominik')
            _, dominik_mr_seg_files = step1_init_data_list.appart_img_and_seg(dominik_all_mr_file_list)
            dominik_segs, dominik_tissue_segs = step1_init_data_list.appart_seg_and_tissueseg(dominik_mr_seg_files)
        
        if mode == 'Synthrad_Anika':
            segs = synthrad_segs+anika_segs
            tissue_segs = synthrad_tissue_segs + anika_tissue_segs
        elif mode == 'Synthrad':
            segs = synthrad_segs
            tissue_segs = synthrad_tissue_segs
        elif mode == 'Anika':
            segs = anika_segs
            tissue_segs = anika_tissue_segs
        elif mode == 'Dominik':
            segs = dominik_segs
            tissue_segs = dominik_tissue_segs

        from dataprocesser import Preprocess_MR_Masks_overlay
        Preprocess_MR_Masks_overlay.main(segs, tissue_segs)

    if  'step4_create_csv' in steps_conducted:
        #ifwritecsv = False if 'step3_merge_segs' in steps_conducted else True
        ifwritecsv = True
        
        if mode == 'Synthrad_Anish':
            data1=r'/gpfs/bwfor/work/ws/hd_qf295-foo/synthrad/Task1/pelvis'
            data2=r'synthrad_conversion/healthy_dissec_helix.csv'
            #data1=r'D:\Projects\data\synthrad\train\Task1\pelvis'
            #data2=r'synthrad_conversion/healthy_dissec_home.csv'
            combined_csv=r'synthrad_conversion/combined_new_home.csv'
            step1_init_data_list.combine_lists_synthrad_anish(data1, data2, combined_csv)

        elif mode == 'Synthrad':
            synthrad_dir=r'E:\Projects\yang_proj\data\synthrad\Task1_val\pelvis'
            mr_csv=r'synthrad_conversion/combined_mr_synthrad_val_newserver.csv'
            step1_init_data_list.create_csv_synthrad_mr(synthrad_dir, mr_csv)
        elif mode == 'Anika':
            ct_dir = r'E:\Projects\yang_proj\data\anika\CT'
            mri_dir = r'E:\Projects\yang_proj\data\anika\MR_registrated'
            ct_csv=r'synthrad_conversion/combined_anika_ct_newserver.csv'
            mr_csv=r'synthrad_conversion/combined_anika_mr_newserver.csv'
            step1_init_data_list.create_csv_Anika(ct_dir, mri_dir, ct_csv, mr_csv)

        elif mode =='Synthrad_Anika':
            data1=r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis'
            data2 = r'E:\Projects\yang_proj\data\anika\MR_registrated'
            combined_csv=r'synthrad_conversion/combined_mr_synthrad_anika_newserver.csv'
            combined_list = step1_init_data_list.combine_lists_synthrad_anika_mr(data1, data2, combined_csv, ifwritecsv)
            
        elif mode == 'Dominik':
            mri_dir = r'E:\Projects\yang_proj\data\dominik\Dominik_MR_VIBE'
            mr_csv=r'synthrad_conversion/combined_mr_dominik_newserver.csv'
            step1_init_data_list.create_csv_Dominik(mri_dir, mr_csv)

    if 'step_optional_slicing' in steps_conducted: # optinal step of slicing 
        from dataprocesser import step4_save_intermediate_dataset as slicing
        slicing.run()

    if 'step_optional_view_distribution' in steps_conducted:
        from dataprocesser import Preprocess_MR_Mask_generation as preprocess_mr
        patient_ID = '1PA010'
        preprocess_mr.analyse_hist(os.path.join('E:\Projects\yang_proj\data\synthrad\Task1\pelvis', patient_ID, 'mr.nii.gz'))

    if 'step5_preprocess' in steps_conducted: # step test data
        from dataprocesser import Preprocess_MR_Mask_generation as preprocess_mr
        
        body_threshold = 30
        if mode=='Synthrad_Anika':
            csv_file = r'synthrad_conversion/combined_mr_synthrad_newserver.csv'
            output_folder1 = r'E:\Projects\yang_proj\data\anika\MR_processed'
            output_folder2 = r'E:\Projects\yang_proj\data\anika\MR_seg_processed'
            csv_simulation_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_T1T2.csv'
            output_mr_csv_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\combined_mr_processed_csv_file.csv'
            preprocess_mr.process_csv(csv_file, output_folder1, output_folder2, csv_simulation_file, body_threshold, output_mr_csv_file)
        if mode=='Synthrad':
            csv_file = r'synthrad_conversion/combined_mr_synthrad_val_newserver.csv'
            output_folder1 = r'E:\Projects\yang_proj\data\synthrad\Task1_val\MR_processed'
            output_folder2 = r'E:\Projects\yang_proj\data\synthrad\Task1_val\MR_seg_processed'
            csv_simulation_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_T1T2.csv'
            output_mr_csv_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\combined_mr_synthrrad_valprocessed.csv'
            preprocess_mr.process_csv(csv_file, output_folder1, output_folder2, csv_simulation_file, body_threshold, output_mr_csv_file)
        elif mode=='Dominik':
            csv_file = r'synthrad_conversion/combined_mr_dominik_newserver.csv'
            output_folder1 = r'E:\Projects\yang_proj\data\dominik\MR_processed'
            output_folder2 = r'E:\Projects\yang_proj\data\dominik\MR_seg_processed'
            csv_simulation_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_T1T2.csv'
            output_mr_csv_file = r'synthrad_conversion/combined_mr_dominik_processed_csv_file.csv'
            preprocess_mr.process_csv(csv_file, output_folder1, output_folder2, csv_simulation_file, body_threshold, output_mr_csv_file)

    if 'step6_testdata' in steps_conducted: # step test data
        from dataprocesser import step6_data_test as data_test
        data_test.run()

    if 'step7_savemask' in steps_conducted: # step for evaluation
        from dataprocesser import step7_save_mask as save_mask
        save_mask.run()
    
    if 'TrainOrTest' in steps_conducted:
        from synthrad_conversion import train
        train.run(input_args, dataset_name = 'combined_simplified_csv_seg_mr_loader')

    if 'step_optional_take_preprocess_example' in steps_conducted:
        dataset_folder=r'E:\Projects\yang_proj\data\anika\MR_registrated'
        orig_mr_file=os.path.join(dataset_folder, r'moved_mr-volume-42553013_t1_vibe_opp_tra - 3.nii.gz')
        seg_file=os.path.join(dataset_folder, r'moved_mr-volume-42553013_t1_vibe_opp_tra - 3_seg.nii.gz')
        tissue_seg_file=os.path.join(dataset_folder, r'moved_mr-volume-42553013_t1_vibe_opp_tra - 3_seg_tissue.nii.gz')
        #merged_seg_file=
        #normalized_mri=
        #contour_file=
        #processed_mr_file=
        #simulated_mr_mask=
        output_folder = r'logs'
        input_path = orig_mr_file
        import nrrd
        from dataprocesser.Preprocess_MR_Mask_generation import normalize
        import numpy as np
        from PIL import Image
        if input_path.endswith('.nrrd'):
            img, header = nrrd.read(input_path)
        elif input_path.endswith('.nii.gz'):
            import nibabel as nib
            img_metadata = nib.load(input_path)
            img = img_metadata.get_fdata()
            affine = img_metadata.affine
        
        # 归一化处理
        norm_max=255 #255
        low_percentile = 5
        high_percentile = 90
        img_normalized = normalize(img, 0, norm_max, np.percentile(img, low_percentile), np.percentile(img, high_percentile), epsilon=0)
        # Choose the slice (e.g., middle slice along the first axis)
        slice_idx = img_normalized.shape[0] // 2  # Change this index based on which slice you want
        slice_img = img_normalized[slice_idx, :, :]  # Extract slice

        # Convert the slice to an image and save as PNG
        #output_path = r'logs/slice_image.png'
        #Image.fromarray(slice_img).save(output_path)

        output_path = r'logs/moved_mr-volume-42553013_t1_vibe_opp_tra - 3_normalized.nii.gz'
        if input_path.endswith('.nrrd'):
            # 保存处理后的 MR 图像
            nrrd.write(output_path, img_normalized, header)
            

        elif input_path.endswith('.nii.gz'):
            img_processed = nib.Nifti1Image(img_normalized, affine)
            nib.save(img_processed, output_path)

if __name__ == "__main__":
    run()