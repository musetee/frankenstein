import os
import numpy as np
from torch.utils.data import DataLoader
def finalcheck(train_ds, val_ds, 
               train_volume_ds, val_volume_ds,
               train_loader, val_loader,
               train_patch_ds, 
               train_batch_size, val_batch_size,
               saved_name_train, saved_name_val, 
               indicator_A, indicator_B,
               ifsave=False, ifcheck=False, iftest_volumes_pixdim=False):
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val, indicator_A, indicator_B)
    if iftest_volumes_pixdim:
        test_volumes_pixdim(train_volume_ds, indicator_A, indicator_B)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds, indicator_A, indicator_B)
        check_batch_data(train_loader,val_loader,
                                train_patch_ds,val_volume_ds,
                                train_batch_size,val_batch_size, 
                                indicator_A, indicator_B)

def test_volumes_pixdim(train_volume_ds, indicator_A, indicator_B):
    train_loader = DataLoader(train_volume_ds, batch_size=1)
    for step, data in enumerate(train_loader):
        mr_data=data[indicator_A]
        ct_data=data[indicator_B]
        
        print(f"source image shape: {mr_data.shape}")
        print(f"source image affine:\n{mr_data.meta['affine']}")
        print(f"source image pixdim:\n{mr_data.pixdim}")

        # target image information
        print(f"target image shape: {ct_data.shape}")
        print(f"target image affine:\n{ct_data.meta['affine']}")
        print(f"target image pixdim:\n{ct_data.pixdim}")

def check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds, indicator_A, indicator_B):
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
            current_name = os.path.basename(os.path.dirname(current_item[indicator_A]))
            print(idx, current_name, 'image:', train_check_data[indicator_A].shape, 'label:', train_check_data[indicator_B].shape)
        except:
            ds_idx = idx * 1
            current_item = train_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item[indicator_A]))
            print('check data error! Check the input data:',current_name)
    print("checked all training data.")

    print('check validation data:')
    idx=0
    for idx in range(len(val_loader)):
        try:
            val_check_data = next(val_iterator)
            ds_idx = idx * 1
            current_item = val_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item[indicator_A]))
            print(idx, current_name, 'image:', val_check_data[indicator_A].shape, 'label:', val_check_data[indicator_B].shape)
        except:
            ds_idx = idx * 1
            current_item = val_ds[ds_idx]
            current_name = os.path.basename(os.path.dirname(current_item[indicator_A]))
            print('check data error! Check the input data:',current_name)
    print("checked all validation data.")

def save_volumes(train_ds, val_ds, saved_name_train, saved_name_val, indicator_A, indicator_B):
    shape_list_train=[]
    shape_list_val=[]
    # use the function of saving information before
    for sample in train_ds:
        name = os.path.basename(os.path.dirname(sample[indicator_A]))
        shape_list_train.append({'patient': name})
    for sample in val_ds:
        name = os.path.basename(os.path.dirname(sample[indicator_A]))
        shape_list_val.append({'patient': name})

    np.savetxt(saved_name_train,shape_list_train,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
    np.savetxt(saved_name_val,shape_list_val,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string

def save_information_all(train_ds, val_ds, saved_name_train, saved_name_val, indicator_A, indicator_B):

    np.savetxt(saved_name_train,train_ds,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
    np.savetxt(saved_name_val,val_ds,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string

def check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size, indicator_A, indicator_B):
    for idx, train_check_data in enumerate(train_loader):
        ds_idx = idx * train_batch_size
        current_item = train_patch_ds[ds_idx]
        print('check train data:')
        print(current_item, 'image:', train_check_data[indicator_A].shape, 'label:', train_check_data[indicator_B].shape)
    
    for idx, val_check_data in enumerate(val_loader):
        ds_idx = idx * val_batch_size
        current_item = val_volume_ds[ds_idx]
        print('check val data:')
        print(current_item, 'image:', val_check_data[indicator_A].shape, 'label:', val_check_data[indicator_B].shape)

def len_patchloader(train_volume_ds,train_batch_size, indicator_A, indicator_B):
    slice_number=sum(train_volume_ds[i][indicator_A].shape[-1] for i in range(len(train_volume_ds)))
    print('total slices in training set:',slice_number)

    import math
    batch_number=sum(math.ceil(train_volume_ds[i][indicator_A].shape[-1]/train_batch_size) for i in range(len(train_volume_ds)))
    print('total batches in training set:',batch_number)
    return slice_number,batch_number

def get_length(dataset, patch_batch_size, indicator_A, indicator_B):
    loader=DataLoader(dataset, batch_size=1)
    iterator = iter(loader)
    sum_nslices=0
    for idx in range(len(loader)):
        check_data = next(iterator)
        nslices=check_data[indicator_A].shape[-1]
        sum_nslices+=nslices
    if sum_nslices%patch_batch_size==0:
        return sum_nslices//patch_batch_size
    else:
        return sum_nslices//patch_batch_size+1