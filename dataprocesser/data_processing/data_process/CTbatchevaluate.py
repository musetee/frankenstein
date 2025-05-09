import glob
import os
from CTevaluate import *

def singleevaluate(file_path,  window_width = 150, window_level = 30):
    # Load the NIfTI CT image data using nibabel.
    if file_path.endswith('.nii.gz'):
        ct_image_nifti = nib.load(file_path)
        ct_image_data = ct_image_nifti.get_fdata()
        ct_image_nifti = nib.load(file_path)
        ct_image_data = ct_image_nifti.get_fdata()
    elif file_path.endswith('.nrrd'):
        ct_image_data, header = nrrd.read(file_path)
    
    ct_data_shape=ct_image_data.shape

    #plot_ct_value_distribution(ct_image_data)
    ct_image_data = ct_windowing(ct_image_data, window_width, window_level)
    #plot_ct_value_distribution(ct_image_data)
    
    # cut roi
    center_x = ct_image_data.shape[0] // 2
    center_y = ct_image_data.shape[1] // 2
    ct_image_roi=extract_roi(ct_image_data, center_x=center_x, center_y=center_y, length=300, width=300)
    ct_image_roi_mean=np.mean(ct_image_roi)

    # Calculate contrast and standard deviation of CT values.
    contrast = calculate_contrast(ct_image_data)
    std_deviation = calculate_standard_deviation(ct_image_data)
    return ct_image_roi_mean, contrast, std_deviation, ct_data_shape

def batchevaluate(dataset_path, format='.nii.gz', save_path='', nii_name='test'):
    for patient_data in glob.glob(dataset_path + "/*"):
        if patient_data.endswith(format):
            patient_name=os.path.basename(os.path.normpath(patient_data))
            print('-------------', patient_name, '-------------')
            ct_image_roi_mean, contrast, std_deviation, ct_data_shape = singleevaluate(patient_data)
            with open(os.path.join(save_path, f'{nii_name}.txt'), 'a') as f:
                f.write('-------------'+patient_name+'-------------\n')
                f.write('Mean of CT values in ROI: '+str(ct_image_roi_mean)+'\n')
                f.write('Contrast of CT image: '+str(contrast)+'\n')
                f.write('Standard Deviation of CT values: '+str(std_deviation)+'\n')
                f.write('Size of CT image: '+str(ct_data_shape)+'\n')
def main():
    dataset_path=r'D:\Data\dataNeaotomAlpha\NIFTI23072115'
    batchevaluate(dataset_path=dataset_path, format='.nii.gz', save_path=dataset_path, nii_name='evaluate')

if __name__=="__main__":
    main()