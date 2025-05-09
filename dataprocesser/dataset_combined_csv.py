from dataprocesser.step0_dataset_base import BaseDataLoader
import os
import pandas as pd
from dataprocesser import customized_transform_list
from dataprocesser.customized_transforms import DivideTransformd, NormalizationMultimodal
from monai.transforms import ScaleIntensityd
from dataprocesser.dataset_registry import register_dataset

@register_dataset('multimodal_csv')
def load_multimodal_csv(opt, my_paths):
    return multimodal_csv_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('multimodal_prior_csv')
def load_multimodal_csv(opt, my_paths):
    return multimodal_prior_csv_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('combined_simplified_csv_seg_assigned')
def load_combined_simplified_csv_seg_assigned(opt, my_paths):
    return combined_simplified_csv_seg_assigned_loader(opt,my_paths,dimension=opt.dataset.input_dim, anatomy_list='synthrad_conversion/TA2_CT_from0.csv')

@register_dataset('combined_simplified_csv_seg_without_assigned_loader')
def load_combined_simplified_csv_seg_without_assigned_loader(opt, my_paths):
    return combined_simplified_csv_seg_without_assigned_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('combined_simplified_csv_seg_mr_loader')
def load_combined_simplified_csv_seg_mr_loader(opt, my_paths):
    return combined_simplified_csv_seg_mr_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('mr2ct_simplified_csv')
def load_mr2ct_simplified_csv(opt, my_paths):
    return combined_simplified_csv_seg_mr2ct_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('xcat_ct_simplified_csv')
def load_xcat_ct_simplified_csv(opt, my_paths):
    return combined_simplified_csv_XCAT_loader(opt,my_paths,dimension=opt.dataset.input_dim)

@register_dataset('synthetic_ct_simplified_csv')
def load_synthetic_ct_simplified_csv(opt, my_paths):
    return combined_simplified_csv_synthetic_loader(opt,my_paths,dimension=opt.dataset.input_dim)

def list_img_seg_ad_pIDs_from_new_simplified_csv(csv_file):
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    patient_IDs = data_frame.iloc[:, 0].tolist()
    Aorta_diss = data_frame.iloc[:, 1].tolist()
    segs =  data_frame.iloc[:, 2].tolist()
    images = data_frame.iloc[:, 3].tolist()
    return patient_IDs, Aorta_diss, segs, images

# for dataset csv of id,Aorta_diss,seg,seg_tissue,img,modality
def list_info_from_multimodal_csv(csv_file):
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    patient_IDs = data_frame.iloc[:, 0].tolist()
    Aorta_diss = data_frame.iloc[:, 1].tolist()
    segs =  data_frame.iloc[:, 2].tolist()
    seg_tissues =  data_frame.iloc[:, 3].tolist()
    images = data_frame.iloc[:, 4].tolist()
    modalities = data_frame.iloc[:, 4].tolist()
    return patient_IDs, Aorta_diss, segs, seg_tissues, images, modalities

class multimodal_csv_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        print('create combined segmentation dataset with assigned value')
        #self.anatomy_list_csv = kwargs.get('anatomy_list', 'synthrad_conversion/TA2_CT_from0.csv')
        #self.anatomy_list_csv_mr = kwargs.get('anatomy_list_mr', 'synthrad_conversion/TA2_T1T2_from0.csv')
        super().__init__(configs, paths, dimension, **kwargs)

    def init_keys(self):
        print('combined segmentation assigned dataset use keys:',[self.indicator_A, self.indicator_B, 'mask'] )
        self.keys = [self.indicator_A, self.indicator_B, 'mask'] # for the body contour of segmentation mask

    def dataframe_to_dict_list(self, df):
        data_dir = self.configs.dataset.data_dir
        return [
            {
                self.indicator_A: os.path.join(data_dir,row['seg'].replace("\\", "/")),
                self.indicator_B: os.path.join(data_dir,row['img'].replace("\\", "/")),
                'seg_tissue': os.path.join(data_dir,row['seg_tissue'].replace("\\", "/")),
                'mask': os.path.join(data_dir,row['img'].replace("\\", "/")),
                'modality': row['modality'],
                'A_paths': os.path.join(data_dir,row['img'].replace("\\", "/")),
                'B_paths':os.path.join(data_dir, row['seg'].replace("\\", "/")),
                'mask_path': os.path.join(data_dir,row['seg'].replace("\\", "/")),
                'Aorta_diss': row['Aorta_diss'],
                'patient_ID': row['id']
            }
            for _, row in df.iterrows()
        ]
    
    def get_dataset_list(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B

        if not os.path.exists(self.configs.dataset.train_csv):
            print('train_csv:', self.configs.dataset.train_csv)
            raise ValueError('you must input a available csv file in simplified form: id, Aorta_diss, seg, img!')
        else:
            print(f'use train csv: {self.configs.dataset.train_csv}')
        if not os.path.exists(self.configs.dataset.test_csv):
            print('test_csv:', self.configs.dataset.test_csv)
            raise ValueError('you must input a available csv file in simplified form: id, Aorta_diss, seg, img!')
        else:
            print(f'use test csv: {self.configs.dataset.test_csv}')

        train_df = pd.read_csv(self.configs.dataset.train_csv)
        test_df = pd.read_csv(self.configs.dataset.test_csv)

        train_patient_IDs = train_df.iloc[:, 0].tolist()
        test_patient_IDs = test_df.iloc[:, 0].tolist()

        self.train_ds = self.dataframe_to_dict_list(train_df)
        self.val_ds = self.dataframe_to_dict_list(test_df)

        import random
        random.shuffle(self.train_ds)
        
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs

    def get_pretransforms(self, transform_list):
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        transform_list.append(NormalizationMultimodal(keys=[indicator_A,indicator_B]))
        return transform_list

class multimodal_prior_csv_loader(multimodal_csv_loader):
    def init_keys(self):
        print('combined segmentation assigned dataset use keys:',[self.indicator_A, self.indicator_B] )
        self.keys = [self.indicator_A, self.indicator_B] # for the body contour of segmentation mask

    def dataframe_to_dict_list(self, df):
        data_dir = self.configs.dataset.data_dir
        return [
            {
                self.indicator_A: os.path.join(data_dir,row['prior'].replace("\\", "/")),
                self.indicator_B: os.path.join(data_dir,row['img'].replace("\\", "/")),
                'modality': row['modality'],
                'A_paths': os.path.join(data_dir,row['prior'].replace("\\", "/")),
                'B_paths': os.path.join(data_dir,row['img'].replace("\\", "/")),
                'Aorta_diss': row['Aorta_diss'],
                'patient_ID': row['id']
            }
            for _, row in df.iterrows()
        ]
    def get_normlization(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        transform_list.append(NormalizationMultimodal(keys=[indicator_A,indicator_B]))
        return transform_list
    
class combined_simplified_csv_seg_assigned_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        print('create combined segmentation dataset with assigned value')
        self.anatomy_list_csv = kwargs.get('anatomy_list', 'synthrad_conversion/TA2_CT_from0.csv')
        super().__init__(configs, paths, dimension, **kwargs)

    def init_keys(self):
        print('combined segmentation assigned dataset use keys:',[self.indicator_A, self.indicator_B, 'mask'] )
        self.keys = [self.indicator_A, self.indicator_B, 'mask'] # for the body contour of segmentation mask

    def get_dataset_list(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B

        if not os.path.exists(self.configs.dataset.train_csv):
            print('train_csv:', self.configs.dataset.train_csv)
            raise ValueError('you must input a available csv file in simplified form: id, Aorta_diss, seg, img!')
        else:
            print(f'use train csv: {self.configs.dataset.train_csv}')
        if not os.path.exists(self.configs.dataset.test_csv):
            print('test_csv:', self.configs.dataset.test_csv)
            raise ValueError('you must input a available csv file in simplified form: id, Aorta_diss, seg, img!')
        else:
            print(f'use test csv: {self.configs.dataset.test_csv}')

        train_patient_IDs, train_Aorta_diss_list, train_source_file_list, train_target_file_list= list_img_seg_ad_pIDs_from_new_simplified_csv(self.configs.dataset.train_csv)
        train_mask_file_list=train_target_file_list

        test_patient_IDs, test_Aorta_diss_list, test_source_file_list, test_target_file_list= list_img_seg_ad_pIDs_from_new_simplified_csv(self.configs.dataset.test_csv)
        test_mask_file_list=test_target_file_list
        # here the original CT images are loaded as mask because they will be further processed as body contour and merged into mask.

        train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, ad, pID in zip(train_source_file_list, train_target_file_list, train_mask_file_list, train_Aorta_diss_list, train_patient_IDs)]
        
        val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, k, ad, pID in zip(test_source_file_list, test_target_file_list, test_mask_file_list, test_Aorta_diss_list, test_patient_IDs)]

        self.train_ds=train_ds
        self.val_ds=val_ds
        self.train_patient_IDs=train_patient_IDs
        self.test_patient_IDs=test_patient_IDs

    def get_pretransforms(self, transform_list):
        transform_list=customized_transform_list.add_CreateContour_MergeMask_MaskHUAssign_transforms(transform_list, self.indicator_A, self.anatomy_list_csv)
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        WINDOW_LEVEL=self.configs.dataset.WINDOW_LEVEL
        WINDOW_WIDTH=self.configs.dataset.WINDOW_WIDTH
        transform_list=customized_transform_list.add_Windowing_ZeroShift_ContourFilter_A_B_transforms(transform_list, WINDOW_LEVEL, WINDOW_WIDTH, self.indicator_A, self.indicator_B)
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        transform_list=customized_transform_list.add_normalization_transform_A_B(transform_list, normalize, indicator_A, indicator_B)
        return transform_list

class combined_simplified_csv_seg_without_assigned_loader(combined_simplified_csv_seg_assigned_loader):
    def get_pretransforms(self, transform_list):
        transform_list=customized_transform_list.add_CreateContour_MergeMask_transforms(transform_list, self.indicator_A)
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        WINDOW_LEVEL=self.configs.dataset.WINDOW_LEVEL
        WINDOW_WIDTH=self.configs.dataset.WINDOW_WIDTH
        transform_list=customized_transform_list.add_Windowing_ZeroShift_ContourFilter_single_B_transforms(transform_list, WINDOW_LEVEL, WINDOW_WIDTH, self.indicator_B)
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        transform_list = customized_transform_list.add_normalization_transform_single_B(transform_list, self.indicator_B, normalize)
        return transform_list

class combined_simplified_csv_seg_mr2ct_loader(combined_simplified_csv_seg_assigned_loader):
    def get_pretransforms(self, transform_list):
        transform_list=customized_transform_list.add_CreateContour_MergeMask_transforms(transform_list, self.indicator_A)
        return transform_list

class combined_simplified_csv_synthetic_loader(combined_simplified_csv_seg_assigned_loader):
    def get_pretransforms(self, transform_list):
        from dataprocesser.customized_transforms import CreateBodyContourTransformd
        transform_list.append(CreateBodyContourTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        )
                                                        )
        return transform_list
    
class combined_simplified_csv_XCAT_loader(combined_simplified_csv_seg_assigned_loader):
    def get_pretransforms(self, transform_list):
        from dataprocesser.customized_transforms import CreateBodyContourTransformd
        transform_list.append(CreateBodyContourTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        )
                                                        )
        return transform_list
    
class combined_simplified_csv_seg_mr_loader(combined_simplified_csv_seg_assigned_loader):
    def get_pretransforms(self, transform_list):
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        return transform_list

    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        if normalize=='norm_mr':
            '''
            source code for MONAI-ScaleIntensity
            mina = arr.min()
            maxa = arr.max()

            if mina == maxa:
                return arr * minv if minv is not None else arr

            norm = (arr - mina) / (maxa - mina)  # normalize the array first
            if (minv is None) or (maxv is None):
                return norm
            return (norm * (maxv - minv)) + minv
            '''

            transform_list.append(ScaleIntensityd(keys=[self.indicator_A], minv=0, maxv=1))  
            transform_list.append(ScaleIntensityd(keys=[self.indicator_B], minv=0, maxv=1))  
            print('use norm_mr for normalization')
        elif normalize=='norm_mr_scale':
            transform_list.append(DivideTransformd(keys=[self.indicator_A], divide_factor=10))  
            transform_list.append(DivideTransformd(keys=[self.indicator_B], divide_factor=255))  
            print('use norm_mr_scale for normalization')
        else: 
            print('please only use the norm_mr method for normlization')
        return transform_list

