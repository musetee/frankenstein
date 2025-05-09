import pandas as pd
import numpy as np
import nibabel as nib
import torch
import os
from tqdm import tqdm
#from dataprocesser.customized_transforms import create_body_contour
from dataprocesser.Preprocess_MR_Mask_generation import process_segmentation
from dataprocesser.Preprocess_CT_Mask_generation import process_CT_segmentation_numpy
import difflib

def find_best_match_smart(organ_name, target_names):
    # 完全匹配
    if organ_name in target_names:
        return organ_name
    # 精确 startswith 匹配（如 vertebrae → vertebrae_Lx）
    matches = [t for t in target_names if t.startswith(organ_name)]
    if matches:
        return matches[0]
    # 再 fallback 到 difflib，但严格一些
    import difflib
    closes = difflib.get_close_matches(organ_name, target_names, n=1, cutoff=0.8)
    return closes #[0] if close else None


def convert_segmentation_mask_torch(source_mask, source_csv, target_csv, body_contour_value=1):
    """
    Converts segmentation mask values from source modality to target modality based on organ name mapping.

    Parameters:
    - source_mask (torch.Tensor): The source segmentation mask tensor.
    - source_csv (str): Path to the CSV file of the source modality (CT or MR).
    - target_csv (str): Path to the CSV file of the target modality (MR or CT).
    - body_contour_value (int): The class value for "body contour" in the target modality.

    Returns:
    - target_mask (torch.Tensor): The converted segmentation mask tensor.
    """
    # Load the source and target anatomy lists
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)

    # Create dictionaries mapping class values to organ names
    source_mapping = {}
    for _, row in source_df.iterrows():
        organ_name = row['Organ Name']
        class_value = row.iloc[0]
        source_mapping.setdefault(organ_name, []).append(class_value)

    target_mapping = {}
    for _, row in target_df.iterrows():
        organ_name = row['Organ Name']
        class_value = row.iloc[0]
        target_mapping.setdefault(organ_name, []).append(class_value)

    # Create a reverse mapping from class values to organ names for the source modality
    class_to_organ = {class_value: organ_name for organ_name, class_values in source_mapping.items() for class_value in class_values}

    # Initialize the target mask with zeros
    target_mask = torch.zeros_like(source_mask, dtype=source_mask.dtype)

    # Convert each unique class in the source mask
    unique_classes = torch.unique(source_mask)
    for class_value in unique_classes:
        # Find the corresponding organ name in the source modality
        organ_name = class_to_organ.get(class_value.item(), None)
        
        if class_value.item() == 0:  # Preserve background as is
            target_value = 0
        else:
            # If organ name exists, find the corresponding target class values
            if organ_name and organ_name in target_mapping:
                # Pick the first target class value (or handle overlaps if needed)
                target_value = target_mapping[organ_name][0]
            else:
                # Use body contour class value for unmapped organs
                target_value = body_contour_value
                #print(f'Processing for class {class_value.item()}')
                #print(f'Not found {organ_name} in target mapping, replaced with body contour.')

        # Replace class values in the target mask
        target_mask[source_mask == class_value] = target_value

    return target_mask

def convert_segmentation_mask(source_mask, source_csv, target_csv, body_contour_value=1000):
    """
    Converts segmentation mask values from source modality to target modality based on organ name mapping.

    Parameters:
    - source_mask (ndarray): The source segmentation mask array.
    - source_csv (str): Path to the CSV file of the source modality (CT or MR).
    - target_csv (str): Path to the CSV file of the target modality (MR or CT).
    - body_contour_value (int): The class value for "body contour" in the target modality.

    Returns:
    - target_mask (ndarray): The converted segmentation mask.
    """
    # Load the source and target anatomy lists
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)

    # Create dictionaries mapping class values to organ names and vice versa
    source_mapping = {}
    for _, row in source_df.iterrows():
        organ_name = row['Organ Name']
        class_value = row.iloc[0]
        source_mapping.setdefault(organ_name, []).append(class_value)

    target_mapping = {}
    for _, row in target_df.iterrows():
        organ_name = row['Organ Name']
        class_value = row.iloc[0]
        target_mapping.setdefault(organ_name, []).append(class_value)

    # Create a reverse mapping from class values to organ names for the source modality
    class_to_organ = {class_value: organ_name for organ_name, class_values in source_mapping.items() for class_value in class_values}
    # Initialize the target mask
    target_organ_names = list(target_mapping.keys())
    target_mask = np.full_like(source_mask, 0, dtype=source_mask.dtype)

    print('source mask contains values:', np.unique(source_mask))
    # Convert each unique class in the source mask
    for class_value in np.unique(source_mask):
        # Find the corresponding organ name in the source modality
        organ_name = class_to_organ.get(class_value, None)
        if class_value == 0:
            target_value = 0
        else:
            # If organ name exists, find the corresponding target class values
            if organ_name and organ_name in target_mapping:
                # Pick the first target class value (or handle overlaps if needed)
                target_value = target_mapping[organ_name][0]
            else:
                # Manual mapping: source organ name → target organ name
                manual_mapping = {
                    'intervertebral_discs': 'spinal_cord',
                    'quadriceps_femoris_left':'gluteus_maximus_left',
                    'quadriceps_femoris_right':'gluteus_maximus_right',
                    'thigh_medial_compartment_left': 'gluteus_maximus_left',
                    'thigh_medial_compartment_right': 'gluteus_maximus_right',
                    'thigh_posterior_compartment_left': 'gluteus_maximus_left',
                    'thigh_posterior_compartment_right': 'gluteus_maximus_right',
                    'sartorius_left': 'gluteus_maximus_left',
                    'sartorius_right': 'gluteus_maximus_right',
                    # Add more mappings here as needed
                }
                # Check manual mapping first
                if organ_name in manual_mapping and manual_mapping[organ_name] in target_mapping:
                    matched_name = manual_mapping[organ_name]
                    target_value = target_mapping[matched_name][0]
                    print(f"[Manual match] '{organ_name}' → '{matched_name}' → label {target_value}")
                else:
                    # Fuzzy match fallback
                    if not target_organ_names:
                        raise ValueError("target_organ_names is None or empty!")
                    if not organ_name:
                        raise ValueError("organ_name is None or empty!")
                    close_matches = difflib.get_close_matches(organ_name, target_organ_names, n=1, cutoff=0.4)
                    if close_matches:
                        matched_name = close_matches[0]
                        target_value = target_mapping[matched_name][0]
                        print(f"[Fuzzy match] '{organ_name}' → '{matched_name}' → label {target_value}")
                    else:
                        print(f"[Warning] No match for '{organ_name}', using body contour value.")
                        target_value = body_contour_value
                '''close_matches = difflib.get_close_matches(organ_name, target_organ_names, n=1, cutoff=0.4)
                if close_matches:
                    matched_name = close_matches[0]
                    target_value = target_mapping[matched_name][0]
                    print(f"[Fuzzy match] '{organ_name}' → '{matched_name}' → label {target_value}")
                else:
                    print(f"[Warning] No match for '{organ_name}', using body contour value.")
                    target_value = body_contour_value'''
        # Replace class values in the target mask
        target_mask[source_mask == class_value] = target_value

    return target_mask

def run_mask_conversion(
    mask = r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis\1PA001\ct_seg.nii.gz',
    img = r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis\1PA001\ct.nii.gz',
    MR_csv = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_MR_for_convert.csv',
    CT_csv = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_CT_for_convert.csv',
    output_path = r'mr_mask_from_ct.nii.gz', # output_path = r'ct_mask_from_mr.nii.gz'
    mode = 'ct2mr'
):
    if mode == 'ct2mr':
        body_threshold=-500
        source_csv = CT_csv
        target_csv = MR_csv
    elif mode == 'mr2ct':
        body_threshold=5
        source_csv = MR_csv
        target_csv = CT_csv

    source_mask = mask
    img = img
    
    seg_metadata = nib.load(source_mask)
    seg = seg_metadata.get_fdata()
    affine = seg_metadata.affine

    img_metadata = nib.load(img)
    img = img_metadata.get_fdata()
    affine = img_metadata.affine

    '''body_contour = np.zeros_like(img, dtype=np.int16)
    for i in range(img.shape[2]):
        slice_data = img[:, :, i]
        body_contour[:, :, i] = create_body_contour(slice_data, body_threshold)
    contour = body_contour
    seg_with_contour = seg+contour'''
    seg_with_contour = seg
    target_mask = convert_segmentation_mask(seg_with_contour, source_csv, target_csv, body_contour_value=1)
    if mode == 'ct2mr':
        csv_simulation_file = MR_csv
        csv_values = pd.read_csv(csv_simulation_file, header=None).to_numpy()
        target_mask = process_segmentation(target_mask, csv_values)
    elif mode == 'mr2ct':
        csv_simulation_file = CT_csv
        target_mask = process_CT_segmentation_numpy(target_mask, csv_simulation_file)
    img_processed = nib.Nifti1Image(target_mask, affine)
    nib.save(img_processed, output_path)


def run_mask_conversion_synthrad_test(synthrad_root = r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis', patient_list=['1PA001'], mode = 'ct2mr', output_csv_file = 'ct2mr_conversion.csv'):
    dataset_list = []
    for patient in tqdm(patient_list):
        mr_mask = os.path.join(synthrad_root, patient, 'mr_merged_seg.nii.gz')
        mr_img = os.path.join(synthrad_root, patient, 'mr.nii.gz')
        ct_mask = os.path.join(synthrad_root, patient, 'ct_seg.nii.gz') 
        ct_img = os.path.join(synthrad_root, patient, 'ct.nii.gz')
        MR_csv = r'synthrad_conversion/TA2_MR_for_convert.csv'
        CT_csv = r'synthrad_conversion/TA2_CT_for_convert.csv'
        if mode == 'ct2mr':
            preprocessed_mr_path = r'E:\Projects\yang_proj\data\anika\MR_processed'
            preprocessed_mr_img = os.path.join(preprocessed_mr_path, f'mr_{patient}.nii.gz')
            output_path = os.path.join(synthrad_root, patient, 'mr_mask_from_ct.nii.gz')  
            csv_mr_line = [patient,0,output_path,preprocessed_mr_img]
        
        elif mode == 'mr2ct':
            output_path = os.path.join(synthrad_root, patient, 'ct_mask_from_mr.nii.gz')
            csv_mr_line = [patient,0,output_path,ct_img]
        run_mask_conversion(mr_mask, mr_img, ct_mask, ct_img, MR_csv, CT_csv, output_path, mode)
        dataset_list.append(csv_mr_line)
        
    import csv
    with open(output_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
        csvwriter.writerows(dataset_list) 

def run_mask_conversion_csv(csv_file = r'E:\Projects\yang_proj\data\synthrad\processed\processed_ct_csv_file.csv', mode = 'ct2mr', output_csv_file = 'ct2mr_conversion.csv'):
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    patient_IDs = data_frame.iloc[:, 0].tolist()
    Aorta_diss = data_frame.iloc[:, 1].tolist()
    segs =  data_frame.iloc[:, 2].tolist()
    images = data_frame.iloc[:, 3].tolist()
    aligned_segs = data_frame.iloc[:, 4].tolist()
    dataset_list = []
    synthrad_root = r"E:\Projects\yang_proj\data\synthrad\Task1\pelvis"
    from tqdm import tqdm
    for idx in tqdm(range(len(images))):
        MR_csv = r'synthrad_conversion/TA2_MR_for_convert.csv'
        CT_csv = r'synthrad_conversion/TA2_CT_for_convert.csv'
        patient = patient_IDs[idx]
        if mode == 'ct2mr':
            ct_mask = segs[idx]
            ct_img = images[idx]
            preprocessed_mr_path = r'E:\Projects\yang_proj\data\anika\MR_processed'
            preprocessed_mr_img = os.path.join(preprocessed_mr_path, f'mr_{patient}.nii.gz')

            mr_mask_from_ct_folder = r'E:\Projects\yang_proj\data\synthrad\mr_mask_from_ct'
            output_path = os.path.join(mr_mask_from_ct_folder, f'{patient}_mr_mask_from_ct.nii.gz')   
            csv_mr_line = [patient,0,output_path,preprocessed_mr_img]
            run_mask_conversion(ct_mask, ct_img, MR_csv, CT_csv, output_path, mode)
        
        elif mode == 'mr2ct':
            mr_mask = os.path.join(synthrad_root, patient, 'mr_merged_seg.nii.gz')
            mr_img = os.path.join(synthrad_root, patient, 'mr.nii.gz')
            output_path = os.path.join(synthrad_root, patient, 'ct_mask_from_mr.nii.gz')
            csv_mr_line = [patient,0,output_path,ct_img]
            run_mask_conversion(mr_mask, mr_img, MR_csv, CT_csv, output_path, mode)
        dataset_list.append(csv_mr_line)
        
    import csv
    with open(output_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
        csvwriter.writerows(dataset_list) 

if __name__ == "__main__":
    csv_file = r'E:\Projects\yang_proj\data\synthrad\processed\processed_csv_file.csv'
    mode = 'ct2mr'
    output_csv_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\datacsv\ct2mr_conversion.csv'
    run_mask_conversion_csv(csv_file = csv_file, mode = mode, output_csv_file = output_csv_file)