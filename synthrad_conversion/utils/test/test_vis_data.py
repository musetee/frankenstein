import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
## get images using SimpleITK and plot them
def get_image(data_path,image_idx=0,slice_idx=10, ifprint=True, ifvis=True):
    #list all files in the folder
    file_list=[i for i in os.listdir(data_path) if 'overview' not in i]
        
    # get the target file
    patient_names = file_list[image_idx]
    target_path=os.path.join(data_path,file_list[image_idx])
    target_file = os.listdir(target_path)

    # get the ct file
    ct_file=[i for i in target_file if 'ct' in i]
    ct_file_path=os.path.join(target_path,ct_file[0])
    ct_image=sitk.ReadImage(ct_file_path)
    space_ct = ct_image.GetSpacing()
    ct_array=sitk.GetArrayFromImage(ct_image)

    ''' 
    # test resample
    ct_image_resampled=ppt.resample(ct_file_path,'outputimage.nii.gz',space*2)
    ct_array_resampled=sitk.GetArrayFromImage(ct_image_resampled)
    print('resampled shape:',ct_array_resampled.shape)
    '''
    # get the mask file
    mask_file=[i for i in target_file if 'mask' in i]
    mask_file_path=os.path.join(target_path,mask_file[0])
    mask_image=sitk.ReadImage(mask_file_path)
    mask_array=sitk.GetArrayFromImage(mask_image)
    

    # get the mr file
    mr_file=[i for i in target_file if 'mr' in i]
    mr_file_path=os.path.join(target_path,mr_file[0])
    mr_image=sitk.ReadImage(mr_file_path)
    mr_array=sitk.GetArrayFromImage(mr_image)
    space_mr = mr_image.GetSpacing()
    if ifprint:
        print(file_list[image_idx]) # the first file, 1PA001
        print('spacing of ct image:', space_ct)
        print('spacing of mr image:', space_mr)
        print('shape of ct image:', ct_array.shape)
        print('shape of mask image:',mask_array.shape)
        print('shape of mr image:',mr_array.shape)
        # get the min and max value of ct and mr images
        print('min of ct image:',ct_array.min())
        print('max of ct image:',ct_array.max())
        print('min of mr image:',mr_array.min())
        print('max of mr image:',mr_array.max())
    if ifvis:
        # visualzie the images
        plt.figure(figsize=(5,5))
        plt.subplot(3,1,1)
        plt.imshow(ct_array[slice_idx,:,:],cmap='gray')
        plt.subplot(3,1,2)
        plt.imshow(mask_array[slice_idx,:,:],cmap='gray')
        plt.subplot(3,1,3)
        plt.imshow(mr_array[slice_idx,:,:],cmap='gray')
    return ct_array,mask_array,mr_array,space_ct,space_mr,patient_names

# test dataloader by first 50 images
# test dataloader
from monai.utils import first
def test_dataloader_first(train_loader):
    batch_test = first(train_loader)
    ct_test, mr_test = batch_test["image"], batch_test["label"]
    print('ct_test shape:',ct_test.shape)
    
def test_dataloader_2d_enumerate(train_loader):
    batch_num=len(train_loader)
    for i, batch in enumerate(train_loader):
        ct, mr = batch["image"], batch["label"]
        for j in range(10): # 10 is batch size
            image0=ct[j,0,:,:]
            label0=mr[j,0,:,:]
            if (j+1)%10==0: # show one image every 10 images
                plt.figure(figsize=(5,5))
                plt.subplot(1,2,1)
                plt.imshow(image0,cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(label0,cmap='gray')
        if i==2: # show 10 batches
            break

# test dataloader by enumerate()
def test_dataloader_3d_enumerate(train_loader):
    print(len(train_loader))
    batch_num=len(train_loader)
    for i,batch in enumerate(train_loader):
        ct, mr = batch["image"], batch["label"]
        for j in range(2): # 2 is batch size
            image0=ct[j,0,50,:,:]
            label0=mr[j,0,50,:,:]
            plt.figure(figsize=(10,10))
            plt.subplot(1,2,1)
            plt.imshow(image0,cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(label0,cmap='gray')
        if i==2:
            break
def test_dataloader_2d_iter(train_loader):
    # test dataloader in batch
    iter_num=2
    iterator=iter(train_loader)
    for i in range(iter_num):
        batch=next(iterator)
        # get image and label from batch
        image, label = batch["image"], batch["label"]
        print(image.shape)
        print(label.shape)
        for j in range(10): # 2 is batch size
            if (j+1)%10==0:
                image0=image[j,0,:,:]
                label0=label[j,0,:,:]
                plt.figure(figsize=(10,10))
                plt.subplot(1,2,1)
                plt.imshow(image0,cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(label0,cmap='gray')

# test dataloader by iter and next
def test_dataloader_3d_iter(train_loader):
    # test dataloader in batch
    iter_num=2
    iterator=iter(train_loader)
    for i in range(iter_num):
        batch=next(iterator)
        # get image and label from batch
        image, label = batch["image"], batch["label"]
        print(image.shape)
        print(label.shape)
        for j in range(2): # 2 is batch size
            image0=image[j,0,50,:,:]
            label0=label[j,0,50,:,:]
            plt.figure(figsize=(10,10))
            plt.subplot(1,2,1)
            plt.imshow(image0,cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(label0,cmap='gray')

# if main
if __name__ == "__main__":
    data_pelvis_path=r'D:\Projects\Task1\pelvis'
    _,_,_,_,_,_=get_image(data_pelvis_path,0,50,True,True)