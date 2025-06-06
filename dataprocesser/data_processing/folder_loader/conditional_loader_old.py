import monai
import os
import numpy as np

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    NormalizeIntensity,
    ResizeWithPadOrCrop,
    Rotate90,
    DivisiblePad,
    CenterSpatialCrop,
    SqueezeDim,
)
from torch.utils.data import DataLoader
import torch

class condDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels):
        self.image_files = image_files
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.image_files[index], self.labels[index]
    
def get_dataset(data_pelvis_path, 
                train_number, 
                val_number, 
                normalize='zscore', 
                resized_size=(512,512), 
                div_size=(16,16), 
                center_crop=0,
                train_batch_size=8,
                val_batch_size=1):
    #list all files in the folder
    file_list=[i for i in os.listdir(data_pelvis_path) if 'overview' not in i]
    file_list_path=[os.path.join(data_pelvis_path,i) for i in file_list]
    #list all ct and mr files in folder
    ct_file_list=[os.path.join(j,'ct.nii.gz') for j in file_list_path]
    cond_file_list=[os.path.join(j,'ct_slice_cond.csv') for j in file_list_path]
    
    ############# condition data preparation
    train_cond_src = cond_file_list[0:train_number] #{'label': j, 'cond_paths': j}
    val_cond_src = cond_file_list[-val_number:]

    train_cond = []
    val_cond = []
    for cond_file in train_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                train_cond.append(int(row['slice']))  # Assuming each row contains only one value

    for cond_file in val_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                val_cond.append(int(row['slice']))

    #print(len(train_cond))
    #print(len(val_cond))
    num_classes = max(max(train_cond), max(val_cond))
    print('number of classes:', num_classes)
    class_names = [i+1 for i in range(num_classes)]

    ############# image data preparation
    train_data = ct_file_list[0:train_number]
    val_data = ct_file_list[-val_number:]
    print('all files in dataset:',len(file_list))


    train_transforms = get_transforms(normalize,resized_size,div_size,center_crop=center_crop)

    
    shape_list_train = []
    train_ds_2d = []
    shape_list_val = []
    val_ds_2d = []
    all_slices_train=0
    all_slices_val=0
    # Load 2D slices for training
    for sample in train_data:
        train_ds_2d_image = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        #train_ds_2d_image=DivisiblePadd(["image", "label"], (-1,batch_size), mode="minimum")(train_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample))
        num_slices = train_ds_2d_image.shape[-1]
        #(train_ds_2d_image.shape)
        #print(num_slices)
        shape_list_train.append({'patient': name, 'shape': train_ds_2d_image.shape})
        for i in range(num_slices):
            train_ds_2d.append(train_ds_2d_image[:,:,i])
        all_slices_train += num_slices
    print('length of set:', len(train_ds_2d))
    print('a slice example:', train_ds_2d[0].shape)
    
    # Load 2D slices for validation
    for sample in val_data:
        val_ds_2d_image = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        #val_ds_2d_image=DivisiblePadd(["image", "label"], (-1, batch_size), mode="minimum")(val_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample))
        shape_list_val.append({'patient': name, 'shape': val_ds_2d_image.shape})
        num_slices = val_ds_2d_image.shape[-1]
        for i in range(num_slices):
            val_ds_2d.append(val_ds_2d_image[:,:,i])
        all_slices_val += num_slices
    

    train_dataset = monai.data.Dataset(data=train_ds_2d, transform=train_transforms)
    val_dataset = monai.data.Dataset(data=val_ds_2d, transform=train_transforms)
    #print(len(train_dataset))
    
    ############# combine image and condition data
    train_set=condDataset(train_dataset, train_cond)
    val_set=condDataset(val_dataset, val_cond)


    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set,num_workers=0, batch_size=val_batch_size, pin_memory=torch.cuda.is_available())
    #val_volume_ds, 
    return train_loader,val_loader


def get_transforms(normalize,resized_size,div_size,center_crop=0):
    transform_list=[]
    #transform_list.append(LoadImage(image_only=True))
    transform_list.append(EnsureChannelFirst())

    
    if normalize=='zscore':
        transform_list.append(NormalizeIntensity(nonzero=False, channel_wise=True))
        print('zscore normalization')

    elif normalize=='minmax':
        transform_list.append(ScaleIntensity(minv=-1.0, maxv=1.0))
        print('minmax normalization')
    elif normalize=='none':
        print('no normalization')

    transform_list.append(ResizeWithPadOrCrop(spatial_size=resized_size,mode="minimum"))
    transform_list.append(Rotate90(k=3))
    transform_list.append(DivisiblePad(k=div_size, mode="minimum"))
    if center_crop>0:
        transform_list.append(CenterSpatialCrop(roi_size=(-1,-1,center_crop)))
    ''''''
    train_transforms = Compose(transform_list)
    # volume-level transforms for both image and label
    return train_transforms

def get_cond_transforms(num_class=150):
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    return y_pred_trans, y_trans
    
def get_length(dataset, patch_batch_size):
    loader=DataLoader(dataset, batch_size=1)
    iterator = iter(loader)
    sum_nslices=0
    for idx in range(len(loader)):
        check_data = next(iterator)
        nslices=check_data['image'].shape[-1]
        sum_nslices+=nslices
    if sum_nslices%patch_batch_size==0:
        return sum_nslices//patch_batch_size
    else:
        return sum_nslices//patch_batch_size+1

def check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds):
    # use batch_size=1 to check the volumes because the input volumes have different shapes
    train_loader = DataLoader(train_volume_ds, batch_size=1)
    val_loader = DataLoader(val_volume_ds, batch_size=1)
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    print('check training data:')
    idx=0
    for idx in range(len(train_loader)):
        try:
            train_check_data = next(train_iterator)
            ds_idx = idx * 1
            current_item = train_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item['image']))
            print(idx, current_name, 'image:', train_check_data['image'].shape, 'label:', train_check_data['label'].shape)
        except:
            ds_idx = idx * 1
            current_item = train_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item['image']))
            print('check data error! Check the input data:',current_name)
    print("checked all training data.")

    print('check validation data:')
    idx=0
    for idx in range(len(val_loader)):
        try:
            val_check_data = next(val_iterator)
            ds_idx = idx * 1
            current_item = val_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item['image']))
            print(idx, current_name, 'image:', val_check_data['image'].shape, 'label:', val_check_data['label'].shape)
        except:
            ds_idx = idx * 1
            current_item = val_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item['image']))
            print('check data error! Check the input data:',current_name)
    print("checked all validation data.")

def save_volumes(train_ds, val_ds, saved_name_train, saved_name_val):
    shape_list_train=[]
    shape_list_val=[]
    # use the function of saving information before
    for sample in train_ds:
        name = os.path.basename(os.path.dirname(sample['image']))
        shape_list_train.append({'patient': name})
    for sample in val_ds:
        name = os.path.basename(os.path.dirname(sample['image']))
        shape_list_val.append({'patient': name})
    np.savetxt(saved_name_train,shape_list_train,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
    np.savetxt(saved_name_val,shape_list_val,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string

def check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size):
    for idx, train_check_data in enumerate(train_loader):
        ds_idx = idx * train_batch_size
        current_item = train_patch_ds[ds_idx]
        print('check train data:')
        print(current_item, 'image:', train_check_data['image'].shape, 'label:', train_check_data['label'].shape)
    
    for idx, val_check_data in enumerate(val_loader):
        ds_idx = idx * val_batch_size
        current_item = val_volume_ds[ds_idx]
        print('check val data:')
        print(current_item, 'image:', val_check_data['image'].shape, 'label:', val_check_data['label'].shape)

def main():
    dataset_path_razer=r'C:\Users\56991\Projects\Datasets\Task1\pelvis'
    dataset_path_server = r"F:\yang_Projects\Datasets\Task1\pelvis"
    train_loader,val_loader = get_dataset(dataset_path_server, train_number=5, val_number=1)
    from tqdm import tqdm
    parameter_file=r'.\test.txt'
    for image, label in tqdm(train_loader):
         with open(parameter_file, 'a') as f:
            f.write('image batch:' + str(image.shape)+'\n')
            f.write('label batch:' + str(label)+'\n')
            f.write('\n')

if __name__ == '__main__':
    main()
