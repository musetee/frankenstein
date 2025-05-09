from dataprocesser.step0_dataset_base import BaseDataLoader
from dataprocesser.dataset_registry import register_dataset
from dataprocesser.dataset_synthrad import synthrad_seg_loader
from dataprocesser.dataset_anish import anish_seg_loader
from dataprocesser import customized_transform_list
from torch.utils.data import ConcatDataset


@register_dataset('combined')
def load_combined(opt, my_paths):
    return combined_seg_loader(opt, my_paths, dimension=2,
                                            train_number_1=30,
                                            train_number_2=153,
                                            val_number_1=2,
                                            val_number_2=0)

@register_dataset('combined_assigned')
def load_combined_assigned(opt, my_paths):
    return combined_seg_assigned_loader(opt, my_paths, dimension=2,
                                                     train_number_1=2,
                                                     train_number_2=1,
                                                     val_number_1=1,
                                                     val_number_2=0,
                                                     data_dir_1=r'D:\Projects\data\synthrad\train\Task1\segmented',
                                                     data_dir_2=r'synthrad_conversion/healthy_dissec_home.csv')



class combined_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2,**kwargs): 
        print('create combined segmentation dataset')
        self.dimension = dimension
        self.train_number_1 = kwargs.get('train_number_1', 170) 
        self.train_number_2 = kwargs.get('train_number_2', 152)  
        self.val_number_1 = kwargs.get('val_number_1', 10) 
        self.val_number_2 = kwargs.get('val_number_2', 10)  
        self.data_dir_1 = kwargs.get('data_dir_1', 'E:\Projects\yang_proj\data\synthrad\Task1\pelvis')
        self.data_dir_2 = kwargs.get('data_dir_2', 'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\healthy_dissec.csv')
        self.configs=configs
        self.paths=paths
        super().__init__(configs,paths,dimension,**kwargs)
        
    def init_keys(self):
        print('combined segmentation assigned dataset use keys:',[self.indicator_A, self.indicator_B, 'mask'] )
        self.keys = [self.indicator_A, self.indicator_B, 'mask'] # for the body contour of segmentation mask

    def get_dataset_list(self):
        # define the dataset sizes for the dataset 1
        self.configs.dataset.data_dir = self.data_dir_1
        self.configs.dataset.train_number = self.train_number_1
        self.configs.dataset.val_number = self.val_number_1
        self.configs.dataset.source_name = ["ct_seg"]
        self.configs.dataset.target_name = ["ct"]
        self.configs.dataset.offset = 1024
        loader1 = synthrad_seg_loader(self.configs,self.paths,self.dimension)
        source_file_list1 = loader1.source_file_list

        # define the dataset sizes for the dataset 2
        self.configs.dataset.data_dir = self.data_dir_2
        self.configs.dataset.train_number = self.train_number_2
        self.configs.dataset.val_number = self.val_number_2
        self.configs.dataset.offset = 1000
        loader2 = anish_seg_loader(self.configs,self.paths,self.dimension)
        source_file_list2 = loader2.source_file_list

        train_ds1 = loader1.train_ds
        train_ds2 = loader2.train_ds

        val_ds1 = loader1.val_ds
        val_ds2 = loader2.val_ds

        self.train_ds = ConcatDataset([train_ds1, train_ds2])
        self.val_ds = ConcatDataset([val_ds1, val_ds2])
        self.source_file_list = source_file_list1+source_file_list2
    
    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        customized_transform_list.add_CreateContour_MergeMask_transforms(transform_list, indicator_A)
        return transform_list

class combined_seg_assigned_loader(combined_seg_loader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        print('create combined segmentation dataset with assigned value')
        self.anatomy_list_csv = kwargs.get('anatomy_list', 'synthrad_conversion/TA2_anatomy.csv')
        super().__init__(configs, paths, dimension, **kwargs)
        
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
