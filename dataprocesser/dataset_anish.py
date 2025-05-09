from dataprocesser import customized_transform_list
from dataprocesser.step0_dataset_base import BaseDataLoader
from dataprocesser.customized_transforms import (
    MaskHUAssigmentd, 
    CreateMaskWithBonesTransformd)
from dataprocesser.dataset_registry import register_dataset
import os
import csv
import pandas as pd

@register_dataset('anish')
def load_anish(opt, my_paths):
    return anish_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('anish_seg')
def load_anish_seg(opt, my_paths):
    return anish_seg_loader(opt,my_paths,dimension=opt.dataset.input_dim)


def list_img_ad_from_anish_csv(csv_file, load_seg=False):
    images = []
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    images = data_frame.iloc[:, -1].tolist()
    if load_seg:
        images = [filename.replace(".nii", "_seg.nii") for filename in images]
    Aorta_diss = data_frame.iloc[:, -2].tolist()
    return images, Aorta_diss

def list_img_ad_pIDs_from_anish_csv(csv_file, load_seg=False):
    images = []
    patient_IDs = []
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    images = data_frame.iloc[:, -1].tolist()
    if load_seg:
        images = [filename.replace(".nii", "_seg.nii") for filename in images]
    Aorta_diss = data_frame.iloc[:, -2].tolist()

    for idx in range(len(data_frame)):
        img_path = data_frame.iloc[idx, -1]
        patient_ID = os.path.splitext(os.path.basename(img_path))[0]
        patient_IDs.append(patient_ID) 

    return images, Aorta_diss, patient_IDs

def list_img_seg_ad_pIDs_from_anish_csv(csv_file):
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    images = data_frame.iloc[:, -1].tolist()
    segs = [filename.replace(".nii", "_seg.nii") for filename in images]
    Aorta_diss = data_frame.iloc[:, -2].tolist()
    patient_IDs = data_frame.iloc[:, 0].tolist()
    return patient_IDs, Aorta_diss, segs, images


def list_replace_anish_csv(input_csv_file, output_csv_file):
    images = []
    patient_IDs = []
    Aorta_diss_s = []
    data_frame = pd.read_csv(input_csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {input_csv_file}")
    rootpath_new=r'E:\Projects\yang_proj\data\anish'
    rootpath1=r'E:\Datasets\aorta abdomen\aorta abdomen\abdomen_aortic_dissection\abdomen_nifti'
    rootpath2=r'E:\Datasets\aorta abdomen\aorta abdomen\abdomen_healthy_and_AA\abdomen_region'

    all_files_in_new_root = os.listdir(rootpath_new)
    all_file_paths_in_new_root = [os.path.join(rootpath_new, filename) for filename in all_files_in_new_root]
    #print(all_file_paths_in_new_root)
    dataset_list=[]
    for idx in range(len(data_frame)):
        img_path = data_frame.iloc[idx, -1]
        Aorta_diss = data_frame.iloc[idx, -2]
        id = data_frame.iloc[idx, 0]
        if rootpath1 in img_path:
            image=img_path.replace(rootpath1, rootpath_new)
        if rootpath2 in img_path:
            image=img_path.replace(rootpath2, rootpath_new)
        seg=image.replace(".nii", "_seg.nii")
        
        if image not in all_file_paths_in_new_root:
            raise KeyError(f"{image} not in new root")
        elif seg not in all_file_paths_in_new_root:
            raise KeyError(f"{seg} not in new root")
        else:
            pass
        entry = [
                id, 
                Aorta_diss, 
                seg,
                image,
                ]
        dataset_list.append(entry)
        images.append(image)
        Aorta_diss_s.append(Aorta_diss)
        patient_IDs.append(id)

    with open(output_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
        csvwriter.writerows(dataset_list) 
    print('all file paths replaced!')
    return images, Aorta_diss_s, patient_IDs


class anish_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2,**kwargs): 
        print('create anish dataset')
        super().__init__(configs,paths,dimension, **kwargs)

    def get_dataset_list(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number

        
        #source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        #target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        #mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            # check if import data is csv file
            if self.configs.dataset.data_dir.endswith('.csv'):
                csv_file = self.configs.dataset.data_dir
            else:
                raise ValueError('The data directory in this case must be a csv file!')
        else:
            if self.configs.server == 'helix' or self.configs.server == 'helixSingle' or self.configs.server=='helixMultiple':
                csv_file = './healthy_dissec_helix.csv'
                
            else:
                csv_file = './healthy_dissec.csv'
            print('use default csv file:', csv_file)

        source_file_list, source_Aorta_diss_list, patient_IDs=list_img_ad_pIDs_from_anish_csv(csv_file, False)
        target_file_list, _, _ =list_img_ad_pIDs_from_anish_csv(csv_file, False)

        train_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, ad, pID in zip(source_file_list[0:train_number], target_file_list[0:train_number], source_Aorta_diss_list[0:train_number], patient_IDs[0:train_number])]
        val_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, ad, pID in zip(source_file_list[-val_number:], target_file_list[-val_number:], source_Aorta_diss_list[-val_number:], patient_IDs[-val_number:])]
        
        self.train_ds=train_ds
        self.val_ds=val_ds
        self.source_file_list=source_file_list
        self.target_file_list=target_file_list

    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        transform_list.append(CreateMaskWithBonesTransformd(keys=[indicator_A],
                                                tissue_min=self.configs.dataset.tissue_min,
                                                tissue_max=self.configs.dataset.tissue_max,
                                                bone_min=self.configs.dataset.bone_min,
                                                bone_max=self.configs.dataset.bone_max))
        return transform_list
    

class anish_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        print('create anish segmentation mask dataset')
        super().__init__(configs,paths,dimension, **kwargs)

    def init_keys(self):
        print('anish segmentation mask dataset use keys:',[self.indicator_A, self.indicator_B, 'mask'] )
        self.keys = [self.indicator_A, self.indicator_B, 'mask'] # for the body contour of segmentation mask

    def get_dataset_list(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number

        
        #source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        #target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        #mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        print('use csv dataset:',self.configs.dataset.data_dir)
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            # check if import data is csv file
            if self.configs.dataset.data_dir.endswith('.csv'):
                csv_file = self.configs.dataset.data_dir
            else:
                raise ValueError('The data directory in this case must be a csv file!')
        else:
            if self.configs.server == 'helix' or self.configs.server == 'helixSingle' or self.configs.server=='helixMultiple':
                csv_file = './healthy_dissec_helix.csv'
            else:
                csv_file = './healthy_dissec.csv'

        # assume the segmentation files are already created!
        '''
        load_seg=True
        source_file_list, source_Aorta_diss_list, patient_IDs=list_img_ad_pIDs_from_anish_csv(csv_file, load_seg)
        target_file_list, _, _ =list_img_ad_pIDs_from_anish_csv(csv_file)
        mask_file_list, _, _=list_img_ad_pIDs_from_anish_csv(csv_file)
        '''

        patient_IDs,source_Aorta_diss_list,source_file_list,target_file_list=list_img_seg_ad_pIDs_from_anish_csv(csv_file)

        train_patient_IDs=patient_IDs[0:train_number]
        train_Aorta_diss_list=source_Aorta_diss_list[0:train_number]
        train_source_file_list=source_file_list[0:train_number]
        train_target_file_list=target_file_list[0:train_number]
        train_mask_file_list=train_target_file_list

        test_patient_IDs=patient_IDs[-val_number:]
        test_Aorta_diss_list=source_Aorta_diss_list[-val_number:]
        test_source_file_list=source_file_list[-val_number:]
        test_target_file_list=target_file_list[-val_number:]
        test_mask_file_list=test_target_file_list
        # here the original CT images are loaded as mask because they will be further processed as body contour and merged into mask.

        train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, ad, pID in zip(
                        train_source_file_list, 
                        train_target_file_list, 
                        train_mask_file_list, 
                        train_Aorta_diss_list, 
                        train_patient_IDs)]
        
        val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, k, ad, pID in zip(
                    test_source_file_list, 
                    test_target_file_list, 
                    test_mask_file_list, 
                    test_Aorta_diss_list, 
                    test_patient_IDs)]
        
        '''
        print('train_ds: \n')
        for i in train_ds:
            print(i)
            print('\n')
            '''

        self.train_ds=train_ds
        self.val_ds=val_ds
        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=target_file_list
        
    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        # create body contour and merge into the segmentations
        transform_list=customized_transform_list.add_CreateContour_MergeMask_transforms(transform_list, indicator_A)
        return transform_list
