from dataprocesser.dataset_synthrad import synthrad_mr2ct_loader
from dataprocesser.dataset_registry import register_dataset
from dataprocesser.step1_init_data_list import extract_patient_id

def pair_list_from_anika_dataset(ct_dir, mri_dir, mri_mode='t1_vibe_in'):
    import os
    from collections import defaultdict
    # Dictionary to hold matching files
    matched_files = defaultdict(lambda: {'CT': None, 'MRI': None})

    # List and process all CT files
    for filename in os.listdir(ct_dir):
        patient_id = extract_patient_id(filename)
        if patient_id:
            matched_files[patient_id]['CT'] = os.path.join(ct_dir, filename)

    # List and process all MRI files
    for filename in os.listdir(mri_dir):
        patient_id = extract_patient_id(filename)
        if patient_id and mri_mode in filename:
            matched_files[patient_id]['MRI'] = os.path.join(mri_dir, filename)

    # Filter out entries that have both CT and MRI data
    # k is the patient ID, v is the dictionary of CT and MRI paths
    matched_pairs = {k: v for k, v in matched_files.items() if v['CT'] and v['MRI']}

    print('all files in dataset:',len(list(matched_pairs.items())))
    # Example output for the first 5 matched pairs
    #for patient_id, paths in list(matched_pairs.items())[:5]:
    #    print(f"Patient ID: {patient_id}, CT: {paths['CT']}, MRI: {paths['MRI']}")
    return matched_pairs

def all_list_from_anika_dataset(ct_dir, mri_dir, mri_mode='t1_vibe_in'):
    import os
    from collections import defaultdict
    # Dictionary to hold matching files
    matched_files = defaultdict(lambda: {'CT': None, 'MRI': None})

    # List and process all CT files
    for filename in os.listdir(ct_dir):
        patient_id = extract_patient_id(filename)
        if patient_id:
            matched_files[patient_id]['CT'] = os.path.join(ct_dir, filename)

    # List and process all MRI files
    for filename in os.listdir(mri_dir):
        patient_id = extract_patient_id(filename)
        if patient_id and mri_mode in filename:
            matched_files[patient_id]['MRI'] = os.path.join(mri_dir, filename)

    # Filter out entries that have both CT and MRI data
    # k is the patient ID, v is the dictionary of CT and MRI paths
    matched_pairs = {k: v for k, v in matched_files.items()}

    print('all files in dataset:',len(list(matched_pairs.items())))
    # Example output for the first 5 matched pairs
    #for patient_id, paths in list(matched_pairs.items())[:5]:
    #    print(f"Patient ID: {patient_id}, CT: {paths['CT']}, MRI: {paths['MRI']}")
    return matched_pairs

def all_list_from_anika_dataset_include_duplicate(ct_dir, mri_dir):
    import os
    # from collections import defaultdict
    # Dictionary to hold matching files
    # matched_files = defaultdict(lambda: {'CT': None, 'MRI': None})
    ct_file_list=all_list_single_modality_from_anika_dataset_include_duplicate(ct_dir)
    mr_file_list=all_list_single_modality_from_anika_dataset_include_duplicate(mri_dir)
    return ct_file_list, mr_file_list

def all_list_single_modality_from_anika_dataset_include_duplicate(ct_dir):
    import os
    # from collections import defaultdict
    # Dictionary to hold matching files
    # matched_files = defaultdict(lambda: {'CT': None, 'MRI': None})
    ct_file_list=[]
    # List and process all CT files
    for filename in os.listdir(ct_dir):
        if filename.endswith('.nii.gz'):
            ct_file_list.append(os.path.join(ct_dir, filename))
        
    #print('dataset:', ct_file_list)
    print('all files in dataset:',len(ct_file_list))
    return ct_file_list

def extract_patientID_from_Anika_dataset(file_list):
    # Pattern to capture patient ID and scan ID
    pattern = r"(?:ct|mr)-volume-(\d+).* - (\d+)\.nii\.gz"
    pattern = r"moved_mr-volume-(\d+).* - (\d+)[a-zA-Z]?\.nii\.gz"
    pattern = r"(?:ct|mr)-volume-(\d+).* - (\d+)[a-zA-Z]?\.(nii\.gz)"

    # Extract patient ID and scan ID
    
    extracted_data = []
    match_idx = 0
    '''
    for file_name in file_list:
        match = re.search(pattern, file_name)
        if match:
            match_idx += 1
            patient_id = match.group(1)
            scan_id = match.group(2)
            extracted_data.append(f'{patient_id}_{scan_id}') # {"patient_id": patient_id, "scan_id": scan_id}
            print(f'get match {match_idx}: ', file_name, f'{patient_id}_{scan_id}')
    '''

    for filename in file_list:
        # Step 1: Remove the file extension
        filename_no_ext = filename.replace(".nii.gz", "")
        
        # Step 2: Split by '-'
        parts = filename_no_ext.split('-')
        
        # Step 3: Extract patient ID and scan ID
        match_idx += 1
        try:
            patient_id = parts[2].split('_')[0]  # Patient ID is the third part after "mr-volume" or "ct-volume"
            scan_id = parts[-1].split('_')[0].strip()  # Scan ID is after the last hyphen, before any trailing letters
            print(f"Filename: {filename}")
            print(f"Patient ID: {patient_id}, Scan ID: {scan_id}\n")
        except IndexError:
            print(f"Filename {filename} does not match expected format.\n")
        extracted_data.append(f'{patient_id}_{scan_id}') # {"patient_id": patient_id, "scan_id": scan_id}
        print(f'get match {match_idx}: ', filename, f'{patient_id}_{scan_id}')
    return extracted_data

class anika_registrated_mr2ct_loader(synthrad_mr2ct_loader):
    def __init__(self,configs,paths,dimension): 
        super().__init__(configs,paths,dimension)

    def get_dataset_list(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size

        # Conditional dictionary keys based on whether masks are loaded
        keys = [indicator_A, indicator_B]

        ct_dir = r'E:\Datasets\M2olie_Patientdata\CT'
        mri_dir = r'E:\Results\MultistepReg\M2olie_Patientdata\Multistep_network_A\predict'
        
        ct_dir = self.configs.dataset.ct_dir #'E:\Datasets\M2olie_Patientdata\CT'
        mri_dir = self.configs.dataset.mri_dir #'E:\Results\MultistepReg\M2olie_Patientdata\Multistep_network_A\predict'
        matched_pairs = pair_list_from_anika_dataset(ct_dir, mri_dir, self.configs.dataset.mri_mode)
        for patient_id, paths in matched_pairs.items():
            print(f"Patient ID: {patient_id}, CT: {paths['CT']}, MRI: {paths['MRI']}")

        # use the matched pairs to form the dataset
        train_ds = [{indicator_A: paths['MRI'], indicator_B: paths['CT']} for patient_id, paths in list(matched_pairs.items())[:train_number]]
        val_ds = [{indicator_A: paths['MRI'], indicator_B: paths['CT']} for patient_id, paths in list(matched_pairs.items())[-val_number:]]
