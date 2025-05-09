import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Rotate90,
    ResizeWithPadOrCrop,
)
from monai.transforms import SaveImage
import numpy as np
import os
import torch
# save validation images
'''nib.save(
    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
)'''

## some functions for GAN training
# output_train_log: to save training loss log to a text file every epoch
# output_val_log: to save validation metrics to a text file every epoch
import monai
from torch import nn

from torchmetrics import MeanAbsoluteError
from torchmetrics.image import StructuralSimilarityIndexMeasure,PeakSignalNoiseRatio
import numpy as np
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.transforms.utils import allow_missing_keys_mode
# save output images
def group_labels(test_labels):
    size_to_labels = {}
    labels_group=[]
    labels_groups=[]
    group_num=0
    size_of_labels = [test_labels[0]['target'].shape]
    for label in test_labels:
        size = label['target'].shape
        if size == size_of_labels[group_num]:
            labels_group.append(label)
        else:
            group_num+=1
            size_of_labels.append(size)
            labels_groups.append(labels_group)
            labels_group=[]
            labels_group.append(label)
        #print(size)
        #print(group_num)
    labels_groups.append(labels_group)
    return labels_groups,size_of_labels
    
# divide the different patients from val_outputs
def write_nifti(val_outputs, output_dir=f'.\logs', filename='val'):
    labels_groups,size_of_labels=group_labels(val_outputs)
    nun_val_patients=len(labels_groups)
    for i in range(nun_val_patients):
        val_output=labels_groups[i]
        # unsqueeze means add a dimension at the position of 3, and then use cat to combine the slices at this position
        concatenated_outputs = torch.cat([label['target'].unsqueeze(3) for label in val_output], dim=3)
        print(concatenated_outputs.shape)
        SaveImage(output_dir=output_dir, output_postfix=f'{filename}_{i}',resample=True)(concatenated_outputs.detach().cpu())#torch.tensor(concatenated_outputs)

def write_nifti_volume(val_outputs, output_dir=f'.\logs', filename='val'):
        SaveImage(output_dir=output_dir, output_postfix=f'{filename}',resample=True)(val_outputs.detach().cpu())

def reverse_transforms(output_images, orig_images,transforms):
        # reverse the transforms
        output_images.applied_operations = orig_images.applied_operations
        val_output_dict = {"target": output_images[0,:,:,:,:]} #  always set val_batch_size=1
        with allow_missing_keys_mode(transforms):
            reversed_images_dict=transforms.inverse(val_output_dict)
        reversed_images=reversed_images_dict["target"]
        return reversed_images
   
def calculate_ssim(pred, target):
    ssim = StructuralSimilarityIndexMeasure().to(pred.device)
    return ssim(pred, target)

def calculate_mae(pred, target):
    mae = MeanAbsoluteError().to(pred.device)
    return mae(pred, target)

def calculate_psnr(pred, target):
    psnr = PeakSignalNoiseRatio().to(pred.device)
    return psnr(pred, target)


def val_log(epoch, step, gen_image, orig_image, saved_path):
    val_ssim=calculate_ssim(gen_image,orig_image)
    val_mae=calculate_mae(gen_image,orig_image)
    val_psnr=calculate_psnr(gen_image,orig_image)
    print(f"val_ssim: {val_ssim}, val_mae: {val_mae}, val_psnr: {val_psnr}.")
    val_metrices = {'ssim': val_ssim, 'mae': val_mae, 'psnr':val_psnr}
    infer_log_file=os.path.join(saved_path, "infer_log.txt")
    output_val_log(epoch, step, infer_log_file, val_metrices)
    return val_metrices, infer_log_file

def output_val_log(epoch, val_step,val_log_file=r'.\logs\val_log.txt',val_metrices={'ssim': 0, 'mae': 0, 'psnr':0}):
    # Save validation log to a text file every epoch
    ssim=val_metrices['ssim'] if 'ssim' in val_metrices else 0
    mae=val_metrices['mae'] if 'mae' in val_metrices else 0
    psnr=val_metrices['psnr'] if 'psnr' in val_metrices else 0
    with open(val_log_file, 'a') as f: # append mode
        f.write(f'epoch {epoch}, val set {val_step}, SSIM: {ssim}, MAE: {mae}, PSNR: {psnr}\n')

def calculate_reverse_info(untransformed_loader):
    ct_data_list=[]
    mri_data_list=[]
    mean_list_ct=[]
    std_list_ct=[]
    mean_list_mri=[]
    std_list_mri=[]
    ct_shape_list=[]
    mri_shape_list=[]
    untransformed_CT_min_list=[]
    untransformed_CT_max_list=[]
    untransformed_MRI_min_list=[]
    untransformed_MRI_max_list=[]
    # calculate the mean and std of the original data
    for idx, checkdata in enumerate(untransformed_loader):
        untransformed_CT=checkdata['target']
        untransformed_MRI=checkdata['source']

        mean_ct=torch.mean(untransformed_CT.float())
        std_ct=torch.std(untransformed_CT.float())
        mean_list_ct.append(mean_ct)
        std_list_ct.append(std_ct)

        mean_mri=torch.mean(untransformed_MRI.float())
        std_mri=torch.std(untransformed_MRI.float())
        mean_list_mri.append(mean_mri)
        std_list_mri.append(std_mri)

        ct_shape_list.append(untransformed_CT.shape)
        mri_shape_list.append(untransformed_MRI.shape)
        untransformed_CT_min_list.append(torch.min(untransformed_CT))
        untransformed_CT_max_list.append(torch.max(untransformed_CT))
        untransformed_MRI_min_list.append(torch.min(untransformed_MRI))
        untransformed_MRI_max_list.append(torch.max(untransformed_MRI))
        ct_data_list.append(untransformed_CT)
        mri_data_list.append(untransformed_MRI)
        all_reverse_info={"CT_mean":mean_list_ct,
                        "CT_std":std_list_ct,
                        "MRI_mean":mean_list_mri,
                        "MRI_std":std_list_mri,
                        "CT_shape":ct_shape_list,
                        "MRI_shape":mri_shape_list,
                        "CT_min":untransformed_CT_min_list,
                        "CT_max":untransformed_CT_max_list,
                        "MRI_min":untransformed_MRI_min_list,
                        "MRI_max":untransformed_MRI_max_list,
                        "CT_data":ct_data_list,
                        "MRI_data":mri_data_list}
    return all_reverse_info

# Define function to reverse normalization
def reverse_normalize_data(tensor, 
                           mean=None, 
                           std=None, 
                           min_val=None, 
                           max_val=None, 
                           mode='zscore'):
    if mode == 'zscore':
        return tensor * std + mean if mean is not None and std is not None else tensor
    elif mode == 'minmax':
        return (tensor+1) /2 * (max_val - min_val) + min_val if min_val is not None and max_val is not None else tensor
    elif mode == 'inputonlyminmax' or mode == 'none' or mode == 'inputonlyzscore':
        tensor = tensor
    return tensor

# Define function to normalize and reverse normalize
def normalize_data(tensor, mean=None, std=None, min_val=None, max_val=None, mode='zscore'):
    if mode == 'zscore':
        return (tensor - mean) / std if mean is not None and std is not None else tensor
    elif mode == 'minmax': # for minmax to -1 and 1
        return (tensor - min_val) / (max_val - min_val) if min_val is not None and max_val is not None else tensor
    return tensor

def save_val_images(val_outputs,val_slice_num,val_names,epoch,saved_img_folder):
    # save validation images
    if val_outputs.shape[0]==sum(val_slice_num):
        # isolate different patients' data
        # val_data_for_check=val_outputs.clone()
        slice_number=val_slice_num # e.g. [200,200,150,230]
        val_data_list=[]
        check_step=0
        for i in slice_number:
            val_data0=val_outputs[:i,:,:,:]
            val_data_list.append(val_data0)
            # delete the first i rows of val_outputs
            val_outputs = val_outputs.narrow(0,i,val_outputs.size(0)-i)
            # check if the data is isolated correctly
            # assert torch.all(val_data_for_check[0:i]==val_data_list[check_step])
            check_step+=1
        # save validation images
        for i in range(len(val_data_list)):
            #height=self.shape_list_val[i]["shape"][1] #338
            #width=self.shape_list_val[i]["shape"][0] #565
            #original_shape=(height,width)
            file_name=f'pred_{val_names[i]}_epoch_{epoch+1}'
            write_nifti(val_data_list[i],saved_img_folder,file_name)
    else:
        print(val_outputs.shape[0])
        print(sum(val_slice_num))
        print("something wrong with validation set, please check")

def compare_imgs(input_imgs, target_imgs, fake_imgs, 
                 saved_name,
                imgformat='jpg',
                dpi = 500,
                model_name='DDPM',):
    from PIL import Image
    input_imgs = input_imgs.squeeze().cpu().numpy()
    input_imgs = (input_imgs * 255).astype(np.uint8)
    input_imgs = Image.fromarray(input_imgs)

    target_imgs = target_imgs.squeeze().cpu().numpy()
    target_imgs = (target_imgs * 255).astype(np.uint8)
    target_imgs = Image.fromarray(target_imgs)

    fake_imgs = fake_imgs.squeeze().cpu().numpy()
    fake_imgs = (fake_imgs * 255).astype(np.uint8)
    fake_imgs = Image.fromarray(fake_imgs)

    titles = ['MRI', 'CT', model_name]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # 
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0.1)
    plt.margins(0,0)
    # MRI image
    axs[0].imshow(input_imgs, cmap='gray')
    axs[0].set_title(titles[0])
    axs[0].axis('off')
    # CT image
    axs[1].imshow(target_imgs, cmap='gray')
    axs[1].set_title(titles[1])
    axs[1].axis('off')
    # fake image
    axs[2].imshow(fake_imgs, cmap='gray')
    axs[2].set_title(titles[2])
    axs[2].axis('off')
    fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    
    # save individual images
    # save output image individually
    title1 = 'MRI'
    fig_mri = plt.figure() #, figsize=(5, 4))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(input_imgs, cmap='gray')
    plt.savefig(saved_name.replace(f'.{imgformat}',f'_mri.{imgformat}'), format=f'{imgformat}'
                , bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig_mri)
    
    title2 = 'CT'
    fig_ct = plt.figure()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(target_imgs, cmap='gray')
    plt.savefig(saved_name.replace(f'.{imgformat}',f'_ct.{imgformat}'), format=f'{imgformat}'
                , bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig_ct)

    title3 = model_name
    fig_fake = plt.figure()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(fake_imgs, cmap='gray')
    plt.savefig(saved_name.replace(f'.{imgformat}',f'_fake.{imgformat}'), format=f'{imgformat}'
                , bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig_fake)


# Define function to save images
def save_single_image(input_imgs,filename, imgformat, dpi=300):
    plt.figure() #, figsize=(5, 4))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.imshow(input_imgs, cmap='gray')
    plt.savefig(filename, format=f'{imgformat}'
                , bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def arrange_images(input_imgs, 
                   label_imgs,
                   fake_imgs,
                   model_name,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        titles = ['MRI', 'CT', model_name]
        fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        cnt = 0

        #print(gen_imgs[cnt].shape)
        axs[0].imshow(input_imgs, cmap='gray') # 0,0,
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(label_imgs, cmap='gray')
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        axs[2].imshow(fake_imgs, cmap='gray')
        axs[2].set_title(titles[2])
        axs[2].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

# Define function to plot histograms
def plot_histogram(data, title, ax, color='blue', alpha=0.7, 
                   x_lower_limit=-1, x_upper_limit=3, y_lower_limit=0, y_upper_limit=15000):
    #x_lower_limit, x_upper_limit = -100, 300 #-1100, 3000
    #y_lower_limit, y_upper_limit = 0, 15000
    bins = 256
    ax.hist(data.flatten(), bins=bins,range=(x_lower_limit, x_upper_limit), color=color, alpha=alpha)
    ax.set_ylim([y_lower_limit, y_upper_limit])
    ax.set_title(title)
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Frequency')

# Arrange three histograms
def arrange_histograms(original, reversed, saved_name, titles=['original','reversed'], dpi=300, mode='ct'):
    # Plot histograms
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plot_histogram(original, f'Histogram for {titles[0]}', axs[0],color='red')
    plot_histogram(reversed, f'Histogram for {titles[1]}', axs[1],color='green')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name, dpi=dpi)
    plt.close(fig)

# Arrange three histograms
def arrange_3_histograms(source,target, output, saved_name , dpi=300,
                         x_lower_limit=-1, x_upper_limit=3, y_lower_limit=0, y_upper_limit=15000):
    # Plot histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plot_histogram(source, f'Histogram for source', axs[0],color='red',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    plot_histogram(target, f'Histogram for target', axs[1],color='green',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    plot_histogram(output, f'Histogram for output', axs[2],color='blue',
                   x_lower_limit=x_lower_limit, x_upper_limit=x_upper_limit, 
                   y_lower_limit=y_lower_limit, y_upper_limit=y_upper_limit)
    #plot_histogram(transformed, f'Histogram for transformed {mode}', axs[2],color='blue')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name, dpi=dpi)
    plt.close(fig)

    # boxplot
    data = [source.flatten(), target.flatten(), output.flatten()]
    plt.boxplot(data)
    plt.xticks([1, 2, 3], ['Source', 'Target', 'Fake'])
    plt.title('Pixel Value Distribution')
    plt.xlabel('Image Type')
    plt.ylabel('Pixel Values')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name.replace('histogram','boxplot'), dpi=dpi)
    plt.close()


def arrange_4_histograms(real1,fake1, real2, fake2, saved_name , dpi=300):
    # Plot histograms
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    plot_histogram(real1, f'Histogram for real1', axs[0],color='red')
    plot_histogram(fake1, f'Histogram for fake1', axs[1],color='red')
    plot_histogram(real2, f'Histogram for real2', axs[2],color='green')
    plot_histogram(fake2, f'Histogram for fake2', axs[3],color='green')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(saved_name, dpi=dpi)
    plt.close(fig)

# save output images
def sample_images(model, input, label,slice_idx, epoch, batch_i, saved_folder, model_name='model'):
    fake = model(input)
    input_imgs=input.cpu().detach().numpy()
    label_imgs=label.cpu().detach().numpy()
    fake_imgs=fake.cpu().detach().numpy()
    gen_imgs = np.concatenate(
         [[input_imgs[slice_idx,0,:,:].squeeze()], 
          [label_imgs[slice_idx,0,:,:].squeeze()], 
          [fake_imgs[slice_idx,0,:,:].squeeze()]])
    
    if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
    saved_name=os.path.join(saved_folder,f"{epoch}_{batch_i}.jpg")
    
    titles = ['MRI', 'CT', 'Translated']
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    cnt = 0
    for j in range(3):
        #print(gen_imgs[cnt].shape)
        axs[j].imshow(gen_imgs[cnt], cmap='gray')
        axs[j].set_title(titles[j])
        axs[j].axis('off')
        cnt += 1
    fig.savefig(saved_name)
    #plt.show()
    plt.close(fig)

    # save individual images
    # save output image individually
    title1 = 'MRI'
    fig_mri, axs_mri = plt.subplots(1, 1) #, figsize=(5, 4))
    axs_mri.imshow(gen_imgs[0].squeeze(), cmap='gray')
    axs_mri.set_title(title1)
    axs_mri.axis('off')
    fig_mri.savefig(saved_name.replace('.jpg','_mri.jpg'))
    plt.close(fig_mri)

    title2 = 'CT'
    fig_ct, axs_ct = plt.subplots(1, 1)
    axs_ct.imshow(gen_imgs[1].squeeze(), cmap='gray')
    axs_ct.set_title(title2)
    axs_ct.axis('off')
    fig_ct.savefig(saved_name.replace('.jpg','_ct.jpg'))
    plt.close(fig_ct)

    title3 = model_name
    fig_fake, axs_fake = plt.subplots(1, 1)
    axs_fake.imshow(gen_imgs[2].squeeze(), cmap='gray')
    axs_fake.set_title(title3)
    axs_fake.axis('off')
    fig_fake.savefig(saved_name.replace('.jpg','_fake.jpg'))
    plt.close(fig_fake)

def save_images(input_imgs, label_imgs,fake_imgs,
                slice_idx,
                saved_name='./test.jpg', 
                imgformat='jpg',
                dpi = 1000,
                model_name='model'):
        titles = ['MRI', 'CT', model_name]
        fig, axs = plt.subplots(1, 3, figsize=(12, 5)) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        cnt = 0

        #print(gen_imgs[cnt].shape)
        axs[0].imshow(input_imgs[:,:,slice_idx].squeeze(), cmap='gray') # 0,0,
        axs[0].set_title(titles[0])
        axs[0].axis('off')

        axs[1].imshow(label_imgs[:,:,slice_idx], cmap='gray')
        axs[1].set_title(titles[1])
        axs[1].axis('off')

        axs[2].imshow(fake_imgs[:,:,slice_idx].squeeze(), cmap='gray')
        axs[2].set_title(titles[2])
        axs[2].axis('off')
        # save image as png

        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

        # save individual images
        # save output image individually
        title1 = 'MRI'
        fig_mri = plt.figure() #, figsize=(5, 4))
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(input_imgs[:,:,slice_idx].squeeze(), cmap='gray')
        plt.savefig(saved_name.replace(f'.{imgformat}',f'_mri.{imgformat}'), format=f'{imgformat}'
                    , bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig_mri)
        
        title2 = 'CT'
        fig_ct = plt.figure()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(label_imgs[:,:,slice_idx].squeeze(), cmap='gray')
        plt.savefig(saved_name.replace(f'.{imgformat}',f'_ct.{imgformat}'), format=f'{imgformat}'
                    , bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig_ct)

        title3 = model_name
        fig_fake = plt.figure()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(fake_imgs[:,:,slice_idx].squeeze(), cmap='gray')
        plt.savefig(saved_name.replace(f'.{imgformat}',f'_fake.{imgformat}'), format=f'{imgformat}'
                    , bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig_fake)

# save output images
def sample_images2(model, input, label,slice_idx, epoch, batch_i, saved_folder):
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_name=f"{epoch}_{batch_i}.jpg"

    fake = model(input)
    input_imgs=input.cpu().detach().numpy()
    target_imgs=label.cpu().detach().numpy()
    fake_imags=fake.cpu().detach().numpy()
    gen_imgs = np.concatenate(
         [[input_imgs[slice_idx,0,:,:].squeeze()], 
          [target_imgs[slice_idx,0,:,:].squeeze()], 
          [fake_imags[slice_idx,0,:,:].squeeze()]])

    titles = ['MRI', 'CT', 'Translated']
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    cnt = 0
    for j in range(3):
        #print(gen_imgs[cnt].shape)
        axs[j].imshow(gen_imgs[cnt], cmap='gray')
        axs[j].set_title(titles[j])
        axs[j].axis('off')
        cnt += 1
    fig.savefig(os.path.join(saved_folder,saved_name))
    #plt.show()
    plt.close(fig)

def sample_images_3D(model, input, label, epoch, batch_i, saved_folder):
    fake = model(input)
    input_imgs=input.cpu().detach().numpy()
    target_imgs=label.cpu().detach().numpy()
    fake_imags=fake.cpu().detach().numpy()
    try:
        gen_imgs = np.concatenate(
            [[input_imgs[0,0,:,:,50].squeeze()], 
            [target_imgs[0,0,:,:,50].squeeze()], 
            [fake_imags[0,0,:,:,50].squeeze()]])
    except:
        gen_imgs = np.concatenate(
            [[input_imgs[0,0,:,:,10].squeeze()], 
            [target_imgs[0,0,:,:,10].squeeze()], 
            [fake_imags[0,0,:,:,10].squeeze()]])
    titles = ['MRI', 'CT', 'Translated']
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    cnt = 0
    for j in range(3):
        #print(gen_imgs[cnt].shape)
        axs[j].imshow(gen_imgs[cnt], cmap='gray')
        axs[j].set_title(titles[j])
        axs[j].axis('off')
        cnt += 1
    if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
    saved_name=f"{epoch}_{batch_i}.jpg"
    fig.savefig(os.path.join(saved_folder,saved_name))
    #plt.show()
    plt.close(fig)
 