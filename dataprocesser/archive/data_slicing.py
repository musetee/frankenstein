from dataprocesser.step1_init_data_list import init_dataset
import os

loader, opt, my_paths = init_dataset()
path=r'E:\Projects\yang_proj\data\seg2med\seg2med_nifti_2d_343'
train_path=os.path.join(path, 'train')
val_path=os.path.join(path,'val')
os.makedirs(path,exist_ok=True)
os.makedirs(train_path,exist_ok=True)
os.makedirs(val_path,exist_ok=True)

loader.save_slices_nifti_and_csv(train_path, loader.train_volume_ds)
loader.save_slices_nifti_and_csv(val_path, loader.val_volume_ds)