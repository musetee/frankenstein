import os
def rename(folder_path):
    ## rename file as the format pID_type_sliceID
    # 1 delete the prefix, prefix: ct, ct_seg, ct_volume
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)
        parts = filename.split('_')
        if len(parts)>=3:
            patient_ID = parts[-4] + '_' + parts[-3]
            sign_name = parts[-2]
            slice_ID = parts[-1]
            
            if sign_name == 'seg':
                new_filename = f'{patient_ID}_seg_{slice_ID}'
            elif sign_name == 'input':
                new_filename = f'{patient_ID}_seg_{slice_ID}'
            elif sign_name == 'target':
                new_filename = f'{patient_ID}_target_{slice_ID}'
            else: # for synthesized and mask
                new_filename = f'{patient_ID}_{sign_name}_{slice_ID}'
            if not new_filename.endswith('.nii.gz'):
                new_filename += '.nii.gz'
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f'Renamed {old_file_path} to {new_file_path}')
folder_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\20241119_0028_Infer_ddpm2d_seg2med_XCAT_CT_56Models_64slices\saved_outputs\volume_output'
rename(folder_path)