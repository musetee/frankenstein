from step1_init_data_list import (
    list_img_ad_from_anish_csv, 
    list_img_pID_from_synthrad_folder,
    )
def run():
    number = 1
    dataset='anish'
    if dataset=='anish':
        data_dir = 'D:\Projects\SynthRad\synthrad_conversion\healthy_dissec_home.csv'
        target_file_list, _ =list_img_ad_from_anish_csv(data_dir) # a csv_file
    elif dataset=='synthrad':
        data_dir = 'D:\Projects\data\synthrad\train\Task1\pelvis'
        target_file_list, _=list_img_pID_from_synthrad_folder(data_dir, accepted_modalities='ct', saved_name="target_filenames.txt")
    create_segmentation(target_file_list[0: number])

def create_segmentation(dataset_list):
    import nibabel as nib
    try:  
        from totalsegmentator.python_api import totalsegmentator
        for sample in dataset_list:
            input_path=sample
            print(f'create segmentation mask for {input_path}')
            output_path=input_path.replace('.nii','_seg.nii')
            input_img = nib.load(input_path)
            totalsegmentator(input=input_img, output=output_path, task='total', fast=False, ml=True)
            print(f'segmentation mask is saved as {output_path}')
    except:
        print("An exception occurred")