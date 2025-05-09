def all_list_from_dominik_dataset(mri_dir):
    file_list=[]
    import os
    # List and process all CT files
    for filename in os.listdir(mri_dir):
        if filename.endswith('.nii'):
            file_list.append(os.path.join(mri_dir, filename))
    return file_list
