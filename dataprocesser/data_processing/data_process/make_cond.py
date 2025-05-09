import glob
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
def make_cond(dataset_path):
    for patient_folder in tqdm(glob.glob(dataset_path + "/*/")):
        if 'overview' not in patient_folder:
            ct_file=os.path.join(patient_folder,'ct.nii.gz') 
            ct_image_nifti = nib.load(ct_file)
            ct_image_data = ct_image_nifti.get_fdata()
            ct_slice_number=ct_image_data.shape[-1]
            ct_slice_label=np.arange(0,ct_slice_number-1,1)
            # write into csv
            with open(os.path.join(patient_folder, 'ct_slice_cond.csv'), 'w') as f:
                f.write('slice\n')
                for i in range(len(ct_slice_label)):
                    f.write(str(ct_slice_label[i])+'\n')

            mr_file=os.path.join(patient_folder,'mr.nii.gz')
            mr_image_nifti = nib.load(mr_file)
            mr_image_data = mr_image_nifti.get_fdata()
            mr_slice_number=mr_image_data.shape[-1]
            mr_slice_label=np.arange(0,mr_slice_number-1,1)
            # write into csv
            with open(os.path.join(patient_folder, 'mr_slice_cond.csv'), 'w') as f:
                f.write('slice\n')
                for i in range(len(mr_slice_label)):
                    f.write(str(mr_slice_label[i])+'\n')
                    
def main():
    dataset_path=r'F:\yang_Projects\Datasets\Task1\pelvis'
    dataset_path_razer=r'C:\Users\56991\Projects\Datasets\Task1\pelvis'
    make_cond(dataset_path)

if __name__=="__main__":
    main()
