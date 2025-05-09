import csv
from dataprocesser.dataset_anika import (
    all_list_single_modality_from_anika_dataset_include_duplicate, 
    extract_patientID_from_Anika_dataset, 
    all_list_from_anika_dataset_include_duplicate)
from dataprocesser.dataset_synthrad import list_img_pID_from_synthrad_folder
from dataprocesser.dataset_anish import list_img_seg_ad_pIDs_from_anish_csv
from dataprocesser.dataset_dominik import all_list_from_dominik_dataset
from dataprocesser.step1_init_data_list import appart_img_and_seg, appart_merged_seg
from dataprocesser.step1_init_data_list import extract_patient_id

def create_csv_combine_lists_synthrad_anika_mr(synthrad_dir, anika_dir_mr, output_mr_csv_file, ifwrtiecsv=True):
    #synthrad_seg_list, synthrad_pIDs = list_img_pID_from_synthrad_folder(synthrad_dir, ["mr_seg"], None)
    seg_name_pattern = "mr_merged_seg" #r"^mr_merged_seg_\d{1}[A-Z]{2}\d{3}$"
    synthrad_seg_list, synthrad_pIDs = list_img_pID_from_synthrad_folder(synthrad_dir, [seg_name_pattern], None)
    synthrad_mr_list, _ = list_img_pID_from_synthrad_folder(synthrad_dir, ["mr"], None)
    synthrad_Aorta_diss = [0] * len(synthrad_seg_list)
    datalist_synthrad = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(synthrad_pIDs, synthrad_Aorta_diss, synthrad_seg_list, synthrad_mr_list)]

    mr_list = all_list_single_modality_from_anika_dataset_include_duplicate(anika_dir_mr)
    mr_files, mr_seg_files = appart_img_and_seg(mr_list)
    mr_seg_files = appart_merged_seg(mr_seg_files)
    mr_pIDs = extract_patientID_from_Anika_dataset(mr_files)

    mr_Aorta_diss = [0] * len(mr_files)
    datalist_mr = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(mr_pIDs, mr_Aorta_diss, mr_seg_files, mr_files)]

    print('length dataset 1: ', len(datalist_synthrad))
    print('length dataset 2: ', len(datalist_mr))
    dataset_list=datalist_synthrad+datalist_mr
    if ifwrtiecsv:
        create_csv_info_file(dataset_list, output_mr_csv_file) 
    return dataset_list

def create_csv_info_file(dataset_list, output_mr_csv_file):
    with open(output_mr_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
        csvwriter.writerows(dataset_list) 
    
def create_csv_synthrad_mr(synthrad_dir, output_csv_file):
    synthrad_seg_list, synthrad_pIDs = list_img_pID_from_synthrad_folder(synthrad_dir, ["mr_merged_seg"], None)
    synthrad_ct_list, _ = list_img_pID_from_synthrad_folder(synthrad_dir, ["mr"], None)
    synthrad_Aorta_diss = [0] * len(synthrad_seg_list)
    datalist_synthrad = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(synthrad_pIDs, synthrad_Aorta_diss, synthrad_seg_list, synthrad_ct_list)]

    print('length dataset 2: ', len(datalist_synthrad))
    dataset_list=datalist_synthrad
    create_csv_info_file(dataset_list, output_csv_file)

def create_csv_combine_lists_synthrad_anish(synthrad_dir, anish_csv, output_csv_file):
    synthrad_seg_list, synthrad_pIDs = list_img_pID_from_synthrad_folder(synthrad_dir, ["ct_seg"], None)
    synthrad_ct_list, _ = list_img_pID_from_synthrad_folder(synthrad_dir, ["ct"], None)
    synthrad_Aorta_diss = [0] * len(synthrad_seg_list)
    
    #anish_pIDs, anish_Aorta_diss, anish_seg_list, anish_ct_list = list_img_seg_ad_pIDs_from_new_simplified_csv(anish_csv)
    anish_pIDs, anish_Aorta_diss, anish_seg_list, anish_ct_list = list_img_seg_ad_pIDs_from_anish_csv(anish_csv)
    datalist_synthrad = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(synthrad_pIDs, synthrad_Aorta_diss, synthrad_seg_list, synthrad_ct_list)]
    datalist_anish = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(anish_pIDs, anish_Aorta_diss, anish_seg_list, anish_ct_list)]

    print('length dataset 1: ', len(synthrad_ct_list))
    print('length dataset 2: ', len(datalist_synthrad))
    dataset_list=datalist_synthrad+datalist_anish
    create_csv_info_file(dataset_list, output_csv_file)

def create_csv_Anika(ct_dir, mri_dir, output_ct_csv_file, output_mr_csv_file):
    ct_list, mr_list = all_list_from_anika_dataset_include_duplicate(ct_dir, mri_dir)
    ct_files, ct_seg_files = appart_img_and_seg(ct_list)
    ct_pIDs = extract_patientID_from_Anika_dataset(ct_files)
    ct_Aorta_diss = [0] * len(ct_list)
    datalist_ct = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(ct_pIDs, ct_Aorta_diss, ct_seg_files, ct_files)]
    create_csv_info_file(datalist_ct, output_ct_csv_file)

    mr_files, mr_seg_files = appart_img_and_seg(mr_list)
    mr_pIDs = extract_patientID_from_Anika_dataset(mr_files)
    mr_Aorta_diss = [0] * len(mr_files)
    datalist_mr = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(mr_pIDs, mr_Aorta_diss, mr_seg_files, mr_files)]
    create_csv_info_file(datalist_mr, output_mr_csv_file)

def create_csv_Dominik(mri_dir, output_mr_csv_file):
    mr_list = all_list_from_dominik_dataset(mri_dir)
    mr_files, mr_seg_files = appart_img_and_seg(mr_list)
    mr_seg_files = appart_merged_seg(mr_seg_files)
    mr_pIDs = [extract_patient_id(mr_file) for mr_file in mr_files]
    mr_Aorta_diss = [0] * len(mr_files)
    datalist_mr = [[id,Aorta_diss,seg,image] for id,Aorta_diss,seg,image in zip(mr_pIDs, mr_Aorta_diss, mr_seg_files, mr_files)]
    create_csv_info_file(datalist_mr, output_mr_csv_file)