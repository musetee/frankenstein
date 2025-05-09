from torch.utils.data import DataLoader
import numpy as np
import os
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