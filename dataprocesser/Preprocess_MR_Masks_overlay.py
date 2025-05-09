import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  
import nrrd
import numpy as np

def load_nifti_file(input_path):
    if input_path.endswith('.nrrd'):
        data, header = nrrd.read(input_path)
        return data, header
    elif input_path.endswith('.nii.gz') or input_path.endswith(".nii"):
        import nibabel as nib
        img_metadata = nib.load(input_path)
        img = img_metadata.get_fdata()
        affine = img_metadata.affine
        return img, affine

def save_nrrd_file(data, HeaderOrAffine, input_path, save_path):
    #nrrd.write(save_path, data, header)
    if input_path.endswith('.nrrd'):
        nrrd.write(save_path, data, HeaderOrAffine)
        
    elif input_path.endswith('.nii.gz') or input_path.endswith(".nii"):
        import nibabel as nib
        img_processed = nib.Nifti1Image(data, HeaderOrAffine)
        nib.save(img_processed, save_path)

def overlay_images(mask_data, organ_data):
    # Combine the images by adding the pixel values
    organ_data = np.where(organ_data == 1, organ_data + 98, organ_data)
    organ_data = np.where(organ_data == 2, organ_data + 197, organ_data)
    organ_data = np.where(organ_data == 3, organ_data + 296, organ_data)
    combined_data = mask_data + organ_data
    return combined_data

def main(files1, files2, output_folder=None):
    # files is the list including all basic MR segmentations
    # files is the list including all basic MR tissue segmentations
        
    print("preprocess length of seg files: ", len(files1))
    print("preprocess length of tissue seg files: ", len(files2))

    files2 = [file.replace('seg_tissue', 'seg') for file in files2]

    files1 = set(files1)
    files2 = set(files2)

    common_files = files1.intersection(files2)

    from tqdm import tqdm
    for filename in tqdm(common_files):
        if filename.endswith(".nrrd") or filename.endswith(".nii.gz") or filename.endswith(".nii"):
            nrrd_path1 =  filename
            nrrd_path2 = filename.replace('seg', 'seg_tissue')
            
            '''
            if os.path.basename(filename) == 'mr_seg.nii.gz':
                patient_ID = os.path.basename(os.path.dirname(filename))
                output_file_name = os.path.basename(filename).replace("seg", f"seg_{patient_ID}") 
            else:
                output_file_name = os.path.basename(filename)
                '''
            
            output_file_name = os.path.basename(filename)
            output_file_name = output_file_name.replace("seg", "merged_seg")
            if output_folder == None:
                output_folder_current_patient = os.path.dirname(filename)
            else:
                output_folder_current_patient = output_folder
            save_path = os.path.join(output_folder_current_patient, output_file_name)

            print(f"Processing {nrrd_path1} and {nrrd_path2}, saving to {save_path}")

            data1, header1 = load_nifti_file(nrrd_path1)
            data2, header2 = load_nifti_file(nrrd_path2)
            
            combined_data = overlay_images(data1, data2)
            save_nrrd_file(combined_data, header1, nrrd_path1, save_path)
            