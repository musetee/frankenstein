from dataprocesser import customized_transform_list
from dataprocesser.step0_dataset_base import BaseDataLoader
from dataprocesser.dataset_registry import register_dataset
from dataprocesser.step0_dataset_base import BaseDataLoader
import os

from monai.transforms import (
    ScaleIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
)

@register_dataset('synthrad_mr2ct')
def load_synthrad_mr2ct(opt, my_paths):
    return synthrad_mr2ct_loader(opt,my_paths,dimension=3)
@register_dataset('synthrad_seg')
def load_synthrad_seg(opt, my_paths):
    return synthrad_seg_loader(opt,my_paths,dimension=opt.dataset.input_dim)


def list_img_pID_from_synthrad_folder(dir, accepted_modalities = ["ct"], saved_name="source_filenames.txt"):
    def is_image_file(filename):
        IMG_EXTENSIONS = [
                '.nrrd', '.nii.gz'
            ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    # it works for root path of any layer:
    # data_path/Task1 or Task2/pelvis or brain
            # |-patient1
            #   |-ct.nii.gz
            #   |-mr.nii.gz
            # |-patient2
            #   |-ct.nii.gz
            #   |-mr.nii.gz
    images = []
    patient_IDs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for roots, _, files in sorted(os.walk(dir)): # os.walk digs all folders and subfolders in all layers of dir
        for file in files:
            if is_image_file(file) and file.split('.')[0] in accepted_modalities:
                path = os.path.join(roots, file)
                patient_ID = os.path.basename(os.path.dirname(path))
                images.append(path)
                patient_IDs.append(patient_ID)
    print(f'Found {len(images)} {accepted_modalities} files in {dir} \n')
    if saved_name is not None:
        with open(saved_name,"w") as file:
            for image in images:
                file.write(f'{image} \n')
    return images, patient_IDs


class synthrad_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2,**kwargs): 
        print('create synthrad segmentation mask dataset')
        super().__init__(configs,paths,dimension, **kwargs)
        
    def init_keys(self):
        print('synthrad segmentation mask dataset use keys:',[self.indicator_A, self.indicator_B, 'mask'] )
        self.keys = [self.indicator_A, self.indicator_B, 'mask'] # for the body contour of segmentation mask

    def get_dataset_list(self):
        # volume-level transforms for both image and label
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        # Conditional dictionary keys based on whether masks are loaded
        
        #list all files in the folder
        file_list=[i for i in os.listdir(self.configs.dataset.data_dir) if 'overview' not in i]
        file_list_path=[os.path.join(self.configs.dataset.data_dir,i) for i in file_list]
        #list all ct and mr files in folder
        
        
        # mask file means the images are used for extracting body contour, see get_pretransforms() below
        source_file_list, patient_IDs=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.source_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"source_filenames.txt"))
        target_file_list, _=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"target_filenames.txt"))
        mask_file_list, _=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"mask_filenames.txt"))


        train_patient_IDs=patient_IDs[0:train_number]
        train_source_file_list=source_file_list[0:train_number]
        train_target_file_list=target_file_list[0:train_number]
        train_mask_file_list=train_target_file_list

        test_patient_IDs=patient_IDs[-val_number:]
        test_source_file_list=source_file_list[-val_number:]
        test_target_file_list=target_file_list[-val_number:]
        test_mask_file_list=test_target_file_list
        # here the original CT images are loaded as mask because they will be further processed as body contour and merged into mask.

        Manual_Set_Aorta_Diss = 0
        ad = Manual_Set_Aorta_Diss

        train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, pID in zip(
                        train_source_file_list, 
                        train_target_file_list, 
                        train_mask_file_list, 
                        train_patient_IDs)]
        
        val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, k, pID in zip(
                    test_source_file_list, 
                    test_target_file_list, 
                    test_mask_file_list, 
                    test_patient_IDs)]
        
        '''train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, pID in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number], patient_IDs[0:train_number])]
        val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, k, pID in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:], patient_IDs[-val_number:])]'''
        
        self.train_ds=train_ds
        self.val_ds=val_ds
        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list

    def get_pretransforms(self, transform_list):
        transform_list=customized_transform_list.add_CreateContour_MergeMask_transforms(transform_list, self.indicator_A)
        return transform_list

class synthrad_mr2ct_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2): 
        super().__init__(configs,paths,dimension)
        
    def get_dataset_list(self):
        # volume-level transforms for both image and label
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        # Conditional dictionary keys based on whether masks are loaded
        
        #list all files in the folder
        #file_list=[i for i in os.listdir(self.configs.dataset.data_dir) if 'overview' not in i]
        # file_list_path=[os.path.join(self.configs.dataset.data_dir,i) for i in file_list]
        #list all ct and mr files in folder
        
        
        source_file_list,pIDs=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.source_name,saved_name=None)
        target_file_list,pIDs=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name,saved_name=None)
        mask_file_list,pIDs=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.mask_name,saved_name=None)

        def write_write_file(images, file):
            with open(file,"w") as file:
                for image in images:
                    file.write(f'{image} \n')
                    
        if self.paths is not None:
            write_write_file(source_file_list, os.path.join(self.paths["saved_logs_folder"],"source_filenames.txt"))
            write_write_file(target_file_list, os.path.join(self.paths["saved_logs_folder"],"target_filenames.txt"))
            write_write_file(mask_file_list, os.path.join(self.paths["saved_logs_folder"],"mask_filenames.txt"))

        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list
    
        train_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'patient_ID': pID } 
                    for i, j , pID in zip(source_file_list[0:train_number], target_file_list[0:train_number], pIDs[0:train_number])]
        val_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'patient_ID': pID} 
                for i, j, pID in zip(source_file_list[-val_number:], target_file_list[-val_number:], pIDs[-val_number:])]
        self.train_ds=train_ds
        self.val_ds=val_ds

    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A, indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')
        elif normalize=='minmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A, indicator_B], minv=-1.0, maxv=1.0))
            print('minmax normalization')

        elif normalize=='scale4000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.99975)) # x=x(1+factor)
            print('scale4000 normalization')

        elif normalize=='scale1000_wrongbutworks':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], factor=-0.999)) 
            print('scale1000 normalization')

        elif normalize=='scale1000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.999)) 
            print('scale1000 normalization')
        
        elif normalize=='scale10':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            #transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None,factor=-0.9)) 
            print('scale10 normalization')

        elif normalize=='inputonlyzscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
            print('only normalize input MRI images')

        elif normalize=='inputonlyminmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=self.configs.dataset.normmin, maxv=self.configs.dataset.normmax))
            print('only normalize input MRI images')
        
        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')
        return transform_list

